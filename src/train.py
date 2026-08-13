import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common import NUM_CLASSES, BATCH_SIZE, DATA_DIR, train_transform, test_transform, build_model, evaluate

# ── 설정 ──
NUM_EPOCHS = 20
LR = 0.001
SAVE_PATH = "checkpoints/baseline.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 모델 로딩
    model = build_model(pretrained_backbone=True)
    model.to(device)

    # 2. 데이터셋 & DataLoader
    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=False, transform=train_transform)
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 4. 학습 루프
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

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

        # 매 epoch마다 검증
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        # best model 저장 — 가장 높은 test 정확도일 때만 덮어씀
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Best model saved ({test_acc:.2f}%)")

    print(f"\nTraining done. Best Test Accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
