import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 아래 common.py 등을 import하기 위함

import torch
import torch_pruning as tp
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from common import BATCH_SIZE, DATA_DIR, test_transform, load_model, evaluate
from export import export_to_onnx, check_onnx_model, verify_parity

# 구조적 프루닝: 채널을 물리적으로 잘라내서 텐서 shape 자체를 줄임(unstructured와 달리
# sparse 포맷/전용 커널 없이도 dense 그대로 사이즈/연산량이 실제로 줄어듦).
# unstructured 실험(src/pruning/unstructured_prune.py)에서 30~50%는 one-shot으로도
# 정확도가 버티는 걸 확인해서, 같은 수준(30%)에서 먼저 확인해봄 — 채널 단위로 통째로
# 잘라내는 거라 같은 비율이라도 unstructured보다 더 타격이 클 수 있어서 낮게 시작.
BASELINE_CHECKPOINT = "checkpoints/baseline.pth"
PRUNED_CHECKPOINT = "checkpoints/pruned_structured.pth"  # fine-tuning 스크립트가 이어받는 입력
OUTPUT_PATH = "checkpoints/pruned_structured.onnx"
PRUNING_RATIO = 0.3
ROUND_TO = 8  # 남는 채널 수를 8의 배수로 맞춤 — 안 맞으면 CPU 벡터 연산(SIMD) 효율이
              # 떨어져서 파라미터가 줄어도 latency가 오히려 늘어나는 현상이 있었음(결과 참고)

if __name__ == "__main__":
    device = torch.device("cpu")
    model = load_model(BASELINE_CHECKPOINT, device)
    model.eval()

    example_inputs = torch.randn(1, 3, 224, 224)

    # 마지막 classifier의 출력 채널(=클래스 수 10)은 고정이라 프루닝 대상에서 제외.
    # 이 레이어의 입력 채널은 앞단이 잘리는 만큼 dependency graph가 알아서 맞춰줌.
    ignored_layers = [model.classifier[1]]

    imp = tp.importance.MagnitudeImportance(p=2)  # L2 norm 기준 채널 중요도
    pruner = tp.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=PRUNING_RATIO,
        global_pruning=True,  # 레이어별 균등이 아니라 전체에서 중요도 낮은 채널부터
        ignored_layers=ignored_layers,
        round_to=ROUND_TO,
    )

    params_before = sum(p.numel() for p in model.parameters())
    pruner.step()
    params_after = sum(p.numel() for p in model.parameters())
    print(f"params: {params_before:,} -> {params_after:,} "
          f"({100 * (1 - params_after / params_before):.1f}% 감소)")

    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    acc = evaluate(model, test_loader, device, desc="structured pruned (fine-tune 전)")
    print(f"accuracy (fine-tune 전): {acc:.2f}%")

    # state_dict가 아니라 모델 객체 통째로 저장 — 구조 자체(채널 수)가 baseline이랑
    # 달라져서 build_model()+load_state_dict()로 재구성이 안 됨. 필요하면 이걸
    # finetune_pruned.py가 torch.load()로 그대로 이어받음.
    torch.save(model, PRUNED_CHECKPOINT)

    model.eval()
    export_to_onnx(model, OUTPUT_PATH)
    check_onnx_model(OUTPUT_PATH)
    is_close, max_diff = verify_parity(model, OUTPUT_PATH)

    print(f"Exported: {OUTPUT_PATH}")
    print("ONNX graph check: OK")
    print(f"PyTorch vs ONNX output match: {'OK' if is_close else 'MISMATCH'} (max diff: {max_diff:.2e})")
