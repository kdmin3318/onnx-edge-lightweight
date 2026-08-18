import copy

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common import BATCH_SIZE, DATA_DIR, train_transform, test_transform, load_model, evaluate

BASELINE_CHECKPOINT = "checkpoints/baseline.pth"
NUM_EPOCHS = 5      # 이미 수렴된 모델을 fine-tuning하는 거라 baseline(20)보다 짧게
LR = 1e-5           # QAT는 fake-quant 노이즈에 적응시키는 거라 baseline(1e-3)보다 훨씬 작은 lr
SAVE_PATH = "checkpoints/qat.pth"

# 최종 배포 타겟이 라즈베리파이(ARM)라서 qnnpack 백엔드로 양자화 시뮬레이션
# (x86 서버에서도 qnnpack으로 돌아가긴 함, 다만 실제 가속은 ARM에서만 의미 있음)
torch.backends.quantized.engine = "qnnpack"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 베이스라인 체크포인트에서 시작
    model = load_model(BASELINE_CHECKPOINT, device)
    model.train()

    # 2. QAT 준비 (FX graph mode) — 모델 안에 fake-quant 노드 삽입
    qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
    example_inputs = (torch.randn(1, 3, 224, 224).to(device),)
    model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
    model.to(device)

    # 3. 데이터셋 & DataLoader
    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=False, transform=train_transform)
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. QAT fine-tuning 루프
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"QAT Epoch [{epoch+1}/{NUM_EPOCHS}]")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        # fake-quant 상태(float)로 정확도 확인 — 실제 int8 변환 후 정확도랑 약간 다를 수 있음
        test_acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)

        print(f"QAT Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            # 실제 int8 양자화 모델로 변환해서 저장 (fake-quant 모델이 아니라 진짜 변환 결과)
            # convert_fx는 CPU 텐서만 지원(qnnpack)하고, 원본 model은 계속 GPU에서
            # 학습을 이어가야 하니 복사본만 CPU로 옮겨서 변환함
            model_to_convert = copy.deepcopy(model).cpu().eval()
            quantized_model = convert_fx(model_to_convert)
            torch.save(quantized_model, SAVE_PATH)  # 구조가 특수해서 state_dict 대신 모델 전체 저장
            print(f"  -> Best QAT model saved ({test_acc:.2f}%)")

    print(f"\nQAT training done. Best Test Accuracy (fake-quant eval): {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
