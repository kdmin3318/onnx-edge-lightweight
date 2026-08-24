import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

NUM_CLASSES = 10
BATCH_SIZE = 64
DATA_DIR = "~/workspace/data/CIFAR10"

# QAT로 만든 int8 양자화 모델(qat.pth)을 실행하려면 필요 (라즈베리파이/ARM 타겟이라 qnnpack)
torch.backends.quantized.engine = "qnnpack"

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


def build_model(pretrained_backbone=True):
    weights = "IMAGENET1K_V1" if pretrained_backbone else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    return model


def load_model(checkpoint_path, device):
    try:
        # qat.pth처럼 TorchScript로 저장된 양자화 모델
        model = torch.jit.load(checkpoint_path, map_location=device)
    except RuntimeError:
        # state_dict만 저장한 경우 (예: baseline.pth)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = build_model(pretrained_backbone=False)
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model


def evaluate(model, loader, device, desc=None):
    model.eval()
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
