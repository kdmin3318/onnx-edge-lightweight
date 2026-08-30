import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 아래 common.py 등을 import하기 위함

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common import BATCH_SIZE, DATA_DIR, train_transform, test_transform, load_model, evaluate
from export import export_to_onnx, check_onnx_model, verify_parity

# structured_prune.py(목표 50%, 실제 76.2% 감소, fine-tune 전 10.00%=랜덤 수준)가 저장한
# pruned 모델을 plain fine-tune 대신 KD로 회복. teacher는 kd_train.py에서 검증한
# resnet50이 아니라 baseline.pth 사용 — 프루닝된 student는 이미 baseline보다도 훨씬
# 낮은 상태에서 시작하므로 baseline 지도만으로 갭 메우기 충분하고, 같은 MobileNetV2
# 구조끼리라 지식 전달도 더 효율적일 것으로 판단(troubleshooting.md §13 참고).
PRUNED_CHECKPOINT = "checkpoints/pruned_structured.pth"
TEACHER_CHECKPOINT = "checkpoints/baseline.pth"
NUM_EPOCHS = 10
LR = 1e-4
ALPHA = 0.3        # hard label(CE) 비중 — 나머지 0.7은 teacher soft label(KLDiv)
TEMPERATURE = 4    # kd_train.py와 동일 설정(teacher가 train set에 매우 확신하는 상태라
                   # 온도로 분포를 부드럽게 펴야 dark knowledge가 드러남)
SAVE_PATH = "checkpoints/pruned_structured_kd_finetuned.pth"
OUTPUT_PATH = "checkpoints/pruned_structured_kd_finetuned.onnx"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = load_model(TEACHER_CHECKPOINT, device)
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
        pbar = tqdm(train_loader, desc=f"KD Fine-tune Epoch [{epoch+1}/{NUM_EPOCHS}]")

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

        print(f"KD Fine-tune Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(student, SAVE_PATH)  # state_dict가 아니라 통째로: 구조 자체가 baseline이랑 다름
            print(f"  -> Best KD fine-tuned model saved ({test_acc:.2f}%)")

    print(f"\nKD fine-tuning done. Best Test Accuracy: {best_accuracy:.2f}%")

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
