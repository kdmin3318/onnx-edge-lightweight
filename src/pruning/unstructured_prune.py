import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 아래 common.py 등을 import하기 위함

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from common import BATCH_SIZE, DATA_DIR, test_transform, load_model, evaluate

# 비구조적(magnitude) 프루닝의 accuracy-vs-sparsity 참고 곡선만 뽑는 스크립트.
# onnxruntime(x86/ARM 둘 다)에 unstructured sparse conv/matmul 커널이 없어서
# 여기서 나오는 latency/size는 baseline과 사실상 동일함 — 그래서 results.csv에는
# 안 넣고 이 스크립트 자체 출력으로만 기록함 (docx/troubleshooting.md 참고).
BASELINE_CHECKPOINT = "checkpoints/baseline.pth"
SPARSITY_LEVELS = [0.3, 0.5, 0.7, 0.9]


def get_prunable_params(model):
    # BatchNorm/bias는 제외하고 Conv2d/Linear의 weight만 대상으로 함
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def compute_sparsity(params):
    total, zeros = 0, 0
    for module, name in params:
        w = getattr(module, name)
        total += w.numel()
        zeros += (w == 0).sum().item()
    return zeros / total


def dense_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def sparse_coo_bytes(tensor):
    # COO 포맷: 0이 아닌 값(values)마다 그 값 자체 + 위치(indices, 차원 수만큼) 저장.
    # PyTorch 기본 인덱스 dtype은 int64(8바이트)라 conv weight(4차원)는 nnz 하나당
    # 인덱스만 4*8=32바이트 — 그래서 sparsity가 충분히 높지 않으면 dense보다 더 커짐.
    sp = tensor.to_sparse()
    values, indices = sp.values(), sp.indices()
    return values.numel() * values.element_size() + indices.numel() * indices.element_size()


if __name__ == "__main__":
    device = torch.device("cpu")
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 프루닝 대상이 아닌 나머지 파라미터(bias, BN 등)는 sparsity와 무관하게 항상 dense로
    # 저장되므로, 체크포인트 전체 예상 사이즈 계산에 쓸 고정 오프셋으로 한 번만 구해둠
    base_model = load_model(BASELINE_CHECKPOINT, device)
    prunable_ids = {id(m) for m, _ in get_prunable_params(base_model)}
    other_bytes = sum(
        p.numel() * p.element_size()
        for m in base_model.modules() if id(m) not in prunable_ids
        for p in m.parameters(recurse=False)
    )

    header = f"{'target':>7} | {'actual':>7} | {'accuracy':>9} | {'dense MB':>9} | {'sparse(COO) MB':>15}"
    print(header)
    for sparsity in SPARSITY_LEVELS:
        model = load_model(BASELINE_CHECKPOINT, device)
        params = get_prunable_params(model)

        # global: 레이어별 개별 기준이 아니라 전체 파라미터를 한 풀에 놓고 L1 크기 기준으로
        # 가장 작은 값들을 sparsity 비율만큼 0으로 만듦 (레이어마다 중요도가 다른 걸 반영)
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)
        for module, name in params:
            prune.remove(module, name)  # mask를 weight에 실제로 곱해 박아넣음 (shape은 그대로)

        actual_sparsity = compute_sparsity(params)
        acc = evaluate(model, test_loader, device, desc=f"sparsity={sparsity:.0%}")

        weight_dense = sum(dense_bytes(getattr(m, n)) for m, n in params)
        weight_sparse = sum(sparse_coo_bytes(getattr(m, n)) for m, n in params)
        dense_mb = (weight_dense + other_bytes) / (1024 * 1024)
        sparse_mb = (weight_sparse + other_bytes) / (1024 * 1024)

        print(f"{sparsity:>6.0%} | {actual_sparsity:>6.2%} | {acc:>8.2f}% | {dense_mb:>8.2f} | {sparse_mb:>14.2f}")
