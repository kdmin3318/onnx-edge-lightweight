import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 아래 common.py 등을 import하기 위함

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common import NUM_CLASSES, BATCH_SIZE, DATA_DIR, train_transform, test_transform, evaluate

# student(MobileNetV2)보다 capacity가 큰 teacher가 필요해서 resnet50 사용.
# baseline.pth(train.py)와 동일한 학습 프로토콜(pretrained backbone + 20 epoch + Adam 1e-3
# + cosine annealing)을 그대로 적용 — teacher가 student보다 나은 이유가 "학습 레시피 차이"가
# 아니라 순수하게 "capacity 차이"만 남도록 하기 위함.
NUM_EPOCHS = 20
LR = 0.001
SAVE_PATH = "checkpoints/teacher_resnet.pth"


def build_teacher(pretrained_backbone=True):
    weights = "IMAGENET1K_V2" if pretrained_backbone else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_teacher(pretrained_backbone=True)
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

        test_acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Best teacher saved ({test_acc:.2f}%)")

    print(f"\nTeacher training done. Best Test Accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
