import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 아래 common.py 등을 import하기 위함

import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torchao.quantization.pt2e import move_exported_model_to_train, move_exported_model_to_eval
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common import BATCH_SIZE, DATA_DIR, train_transform, test_transform, load_model
from inference import measure_latency, log_result

BASELINE_CHECKPOINT = "checkpoints/baseline.pth"
NUM_EPOCHS = 5      # 이미 수렴된 모델을 fine-tuning하는 거라 baseline(20)보다 짧게
LR = 1e-5           # QAT는 fake-quant 노이즈에 적응시키는 거라 baseline(1e-3)보다 훨씬 작은 lr
SAVE_PATH = "checkpoints/qat.pth"

# torch 2.10부터 예전에 쓰던 torch.ao.quantization.quantize_fx(FX graph mode)가 사실상 방치돼서
# convert_fx() 결과가 깨지는 버그가 있었음(모든 이미지를 한 클래스로만 예측). 새로 권장되는
# torch.export 기반 pt2e API(prepare_qat_pt2e/convert_pt2e)로 교체함.
# XNNPACKQuantizer: convert_pt2e() 후 텐서 159개 중 106개가 fp32로 남길래 한때 "quantizer가
# 일부 레이어를 못 알아보는 커버리지 버그"로 의심하고 executorch 공식 버전으로 바꿔봤는데,
# 결과가 토씨 하나 안 다르게 동일해서 그 진단 자체가 틀렸음이 확인됨(docx/troubleshooting.md
# §5) — 106개 중 53개는 원래 fp32로 남는 bias, 나머지 53개는 실제 연산에 안 쓰이는 잔여
# 사본이고 진짜 연산은 별도의 _frozen_paramN(int8) 53개가 담당 — 애초에 전 레이어 100%
# 양자화되고 있었음. executorch 공식/torchao 비공식 quantizer가 동등하다는 게 확인됐으므로
# 이 import는 편의상 유지 중일 뿐, executorch 설치가 필수는 아님.
# 배포용 ONNX 변환은 convert_pt2e() 결과물 대신 QAT로 개선된 fp32 가중치만 뽑아서 별도
# 경로(qat_fp32_export.py)로 처리함 — convert_pt2e()가 만드는 quantized_decomposed 커스텀
# op가 ONNX exporter와 근본적으로 안 맞기 때문(docx/troubleshooting.md §9, §10).


def evaluate_pt2e(model, loader, device, desc=None):
    # pt2e로 export된 모델은 model.eval()/.train() 호출이 금지돼있어서(NotImplementedError)
    # 전용 함수(move_exported_model_to_eval)를 대신 써야 함
    move_exported_model_to_eval(model)
    correct = 0
    total = 0
    iterator = tqdm(loader, desc=desc) if desc else loader
    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 베이스라인 체크포인트에서 시작
    model = load_model(BASELINE_CHECKPOINT, torch.device("cpu"))
    model.eval()

    # 2. torch.export로 그래프 추출. 배치 차원을 동적으로 표시해야 학습(배치=64)/
    # latency 측정(배치=1) 양쪽 다 같은 그래프로 처리 가능함.
    # 주의: 예시 입력 배치를 1로 주면 안 됨 — 크기 1인 차원은 브로드캐스팅 때문에
    # PyTorch가 무조건 고정값으로 취급해버려서 dynamic 표시가 무시됨.
    example_inputs = (torch.randn(BATCH_SIZE, 3, 224, 224),)
    batch_dim = torch.export.Dim("batch", min=1, max=BATCH_SIZE * 2)
    exported = torch.export.export(model, example_inputs, dynamic_shapes=({0: batch_dim},)).module()

    # 3. QAT 준비 — 모델 안에 fake-quant 노드 삽입 (XNNPACK 대상, ARM/모바일 배포용 설정)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
    prepared = prepare_qat_pt2e(exported, quantizer)
    prepared.to(device)

    # 4. 데이터셋 & DataLoader
    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=False, transform=train_transform)
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(prepared.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 6. QAT fine-tuning 루프
    best_accuracy = 0.0
    best_quantized_model = None

    for epoch in range(NUM_EPOCHS):
        move_exported_model_to_train(prepared)
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"QAT Epoch [{epoch+1}/{NUM_EPOCHS}]")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = prepared(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        # fake-quant 상태(float 연산 + 양자화 노이즈 시뮬레이션)로 정확도 확인
        test_acc = evaluate_pt2e(prepared, test_loader, device)
        avg_loss = running_loss / len(train_loader)

        print(f"QAT Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | Test Acc(fake-quant): {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            # 실제 int8 변환. qnnpack/XNNPACK 계열 양자화 커널은 CPU 대상이라 복사본을 CPU로
            # 옮겨서 변환 (원본 prepared는 계속 GPU에서 학습 이어감)
            quantized_model = convert_pt2e(copy.deepcopy(prepared).cpu())
            real_acc = evaluate_pt2e(quantized_model, test_loader, torch.device("cpu"), desc="Real INT8 Eval")
            torch.save(quantized_model.state_dict(), SAVE_PATH)
            best_quantized_model = quantized_model
            print(f"  -> Best QAT model saved (fake-quant {test_acc:.2f}% / 실제 int8 {real_acc:.2f}%)")

    print(f"\nQAT training done. Best fake-quant Test Accuracy: {best_accuracy:.2f}%")

    # inference.py처럼 latency/사이즈까지 측정해서 results.csv에 바로 기록
    # (이 pt2e 양자화 모델은 common.load_model()로 재구성 불가능해서 inference.py CLI로
    # 별도 재실행은 안 됨 — 학습 직후 메모리에 있는 모델로 바로 측정)
    if best_quantized_model is not None:
        latency_ms = measure_latency(best_quantized_model, torch.device("cpu"))
        size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
        print(f"Latency (batch=1): {latency_ms:.2f} ms")
        print(f"Model Size: {size_mb:.2f} MB")
        log_result(SAVE_PATH, real_acc, latency_ms, size_mb)


if __name__ == "__main__":
    main()
