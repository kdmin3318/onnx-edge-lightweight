import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 아래 common.py 등을 import하기 위함

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common import BATCH_SIZE, DATA_DIR, train_transform, test_transform, evaluate
from export import export_to_onnx, check_onnx_model, verify_parity

# structured_prune.py가 저장한, 채널이 이미 잘려나간 모델을 이어받아 일반 fp32로
# 재학습(회복)시킴. QAT(prepare_qat_pt2e)처럼 그래프를 특수 구조로 바꾸는 게 아니라
# 그냥 평범한 nn.Module이라 train.py와 동일한 방식의 학습 루프를 그대로 씀.
PRUNED_CHECKPOINT = "checkpoints/pruned_structured.pth"
NUM_EPOCHS = 10  # fine-tune 전 정확도가 12.73%(사실상 붕괴)라 QAT(5epoch, 노이즈 적응만)보다 더 많은 재학습이 필요할 것으로 보고 늘림
LR = 1e-4        # baseline(1e-3)보다 작지만 QAT(1e-5)보다는 큼 — 잃어버린 capacity를 실제로 복구해야 함
SAVE_PATH = "checkpoints/pruned_structured_finetuned.pth"
OUTPUT_PATH = "checkpoints/pruned_structured_finetuned.onnx"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(PRUNED_CHECKPOINT, map_location="cpu", weights_only=False)
    model.to(device)

    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=False, transform=train_transform)
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch [{epoch+1}/{NUM_EPOCHS}]")

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

        test_acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)
        print(f"Fine-tune Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model, SAVE_PATH)  # state_dict가 아니라 통째로: 구조 자체가 baseline이랑 다름
            print(f"  -> Best fine-tuned model saved ({test_acc:.2f}%)")

    print(f"\nFine-tuning done. Best Test Accuracy: {best_accuracy:.2f}%")

    best_model = torch.load(SAVE_PATH, map_location="cpu", weights_only=False)
    best_model.eval()
    export_to_onnx(best_model, OUTPUT_PATH)
    check_onnx_model(OUTPUT_PATH)
    is_close, max_diff = verify_parity(best_model, OUTPUT_PATH)

    print(f"Exported: {OUTPUT_PATH}")
    print("ONNX graph check: OK")
    print(f"PyTorch vs ONNX output match: {'OK' if is_close else 'MISMATCH'} (max diff: {max_diff:.2e})")


if __name__ == "__main__":
    main()
