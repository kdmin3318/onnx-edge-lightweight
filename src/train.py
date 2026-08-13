import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── 설정 ──
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 2
LR = 0.001
DATA_DIR = "./data"
SAVE_PATH = "best_mobilenetv2_cifar10.pth"

device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 모델 로딩
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.to(device)

#2. 전처리(학습용/평가용 분리)
#학습: augmentation으로 일반화 성능 향상
#평가: 고정된 전처리로 일관된 측정
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

#3. 데이터셋 & DataLoader
train_dataset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
test_dataset = CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#4. Loss / Optimizer / Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

#5. 검증 함수
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

#6. 학습 루프
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

    #매 epoch마다 검증
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
