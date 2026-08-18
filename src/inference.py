import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def evaluate_onnx(session, loader, desc=None):
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    iterator = tqdm(loader, desc=desc) if desc else loader
    for images, labels in iterator:
        outputs = session.run(None, {input_name: images.numpy()})[0]
        predicted = outputs.argmax(axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
    return 100 * correct / total


def measure_latency_onnx(session, input_size=(1, 3, 224, 224), warmup=10, runs=100):
    input_name = session.get_inputs()[0].name
    x = np.random.randn(*input_size).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: x})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: x})
    end = time.perf_counter()
    return (end - start) / runs * 1000  # ms/image


def get_model_size_mb(path):
    total_bytes = os.path.getsize(path)
    external_data_path = path + ".data"  # torch.onnx.export가 가중치를 여기에 분리 저장함
    if os.path.exists(external_data_path):
        total_bytes += os.path.getsize(external_data_path)
    return total_bytes / (1024 * 1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/baseline.pth")
    args = parser.parse_args()

    device = torch.device("cpu")

    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if args.checkpoint.endswith(".onnx"):
        session = ort.InferenceSession(args.checkpoint, providers=["CPUExecutionProvider"])
        accuracy = evaluate_onnx(session, test_loader, desc="Evaluating")
        latency_ms = measure_latency_onnx(session)
    else:
        model = load_model(args.checkpoint, device)
        accuracy = evaluate(model, test_loader, device, desc="Evaluating")
        latency_ms = measure_latency(model, device)

    size_mb = get_model_size_mb(args.checkpoint)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Latency (batch=1): {latency_ms:.2f} ms")
    print(f"Model Size: {size_mb:.2f} MB")
