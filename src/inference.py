import argparse
import os
import time

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from common import BATCH_SIZE, DATA_DIR, test_transform, load_model, evaluate


def measure_latency(model, device, input_size=(1, 3, 224, 224), warmup=10, runs=100):
    x = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        end = time.perf_counter()
    return (end - start) / runs * 1000  # ms/image


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/baseline.pth")
    args = parser.parse_args()

    device = torch.device("cpu")

    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(args.checkpoint, device)

    accuracy = evaluate(model, test_loader, device, desc="Evaluating")
    latency_ms = measure_latency(model, device)
    size_mb = get_model_size_mb(args.checkpoint)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Latency (batch=1): {latency_ms:.2f} ms")
    print(f"Model Size: {size_mb:.2f} MB")
