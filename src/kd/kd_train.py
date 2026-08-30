import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 아래 common.py 등을 import하기 위함

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common import BATCH_SIZE, DATA_DIR, train_transform, test_transform, build_model, evaluate
from train_teacher import build_teacher

# baseline.pth(train.py)와 동일한 프로토콜(pretrained backbone + 20 epoch + Adam 1e-3
# + cosine annealing)로 student를 학습시키되, loss만 teacher(resnet50, 96.18%) 지도를
# 섞은 KD loss로 바꿈 — "학습 레시피는 baseline과 동일한데 teacher 지도가 있으면
# 더 나은가?"만 변수로 남기기 위함.
TEACHER_CHECKPOINT = "checkpoints/teacher_resnet.pth"
NUM_EPOCHS = 20
LR = 0.001
ALPHA = 0.3        # hard label(CE) 비중 — 나머지 0.7은 teacher soft label(KLDiv)
TEMPERATURE = 4    # teacher가 train loss 0.0018까지 내려가 있어서(정답 클래스 확신 ~99.8%)
                   # 그대로 쓰면 dark knowledge가 거의 안 드러남 — T로 분포를 부드럽게 폄
SAVE_PATH = "checkpoints/kd_student.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = build_teacher(pretrained_backbone=False)
    teacher.load_state_dict(torch.load(TEACHER_CHECKPOINT, map_location=device))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = build_model(pretrained_backbone=True)
    student.to(device)

    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=False, transform=train_transform)
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        student.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"KD Epoch [{epoch+1}/{NUM_EPOCHS}]")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            optimizer.zero_grad()
            student_logits = student(images)

            hard_loss = F.cross_entropy(student_logits, labels)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / TEMPERATURE, dim=1),
                F.softmax(teacher_logits / TEMPERATURE, dim=1),
                reduction="batchmean",
            ) * (TEMPERATURE ** 2)
            loss = ALPHA * hard_loss + (1 - ALPHA) * soft_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        test_acc = evaluate(student, test_loader, device)
        avg_loss = running_loss / len(train_loader)

        print(f"KD Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(student.state_dict(), SAVE_PATH)
            print(f"  -> Best KD student saved ({test_acc:.2f}%)")

    print(f"\nKD training done. Best Test Accuracy: {best_accuracy:.2f}% (baseline plain training: 95.42%)")


if __name__ == "__main__":
    main()
