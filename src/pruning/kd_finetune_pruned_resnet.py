import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 아래 common.py 등을 import하기 위함

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common import BATCH_SIZE, DATA_DIR, train_transform, test_transform, evaluate
from export import export_to_onnx, check_onnx_model, verify_parity
from kd.train_teacher import build_teacher

# kd_finetune_pruned.py(teacher=baseline.pth, 90.40%)와 완전히 동일한 조건(같은 76.2%
# 프루닝 체크포인트, α=0.3, T=4, 10 epoch, lr=1e-4)에서 teacher만 resnet50(96.18%)으로
# 교체 — "teacher 자체 성능이 더 좋은 게 이기는지, 같은 구조끼리의 전달 효율이 이기는지"
# 비교하기 위함(troubleshooting.md §14 참고).
PRUNED_CHECKPOINT = "checkpoints/pruned_structured.pth"
TEACHER_CHECKPOINT = "checkpoints/teacher_resnet.pth"
NUM_EPOCHS = 10
LR = 1e-4
ALPHA = 0.3
TEMPERATURE = 4
SAVE_PATH = "checkpoints/pruned_structured_kd_resnet_finetuned.pth"
OUTPUT_PATH = "checkpoints/pruned_structured_kd_resnet_finetuned.onnx"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = build_teacher(pretrained_backbone=False)
    teacher.load_state_dict(torch.load(TEACHER_CHECKPOINT, map_location=device))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = torch.load(PRUNED_CHECKPOINT, map_location="cpu", weights_only=False)
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
        pbar = tqdm(train_loader, desc=f"KD(resnet) Fine-tune Epoch [{epoch+1}/{NUM_EPOCHS}]")

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

        print(f"KD(resnet) Fine-tune Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(student, SAVE_PATH)
            print(f"  -> Best KD(resnet) fine-tuned model saved ({test_acc:.2f}%)")

    print(f"\nKD(resnet) fine-tuning done. Best Test Accuracy: {best_accuracy:.2f}%")

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
