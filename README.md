# onnx-edge-lightweight

MobileNetV2 + CIFAR-10 기반 모델 경량화 파이프라인 실험.
QAT(Quantization-Aware Training)와 ONNX 변환을 통해 엣지 디바이스(라즈베리파이) 배포까지 다룬다.

## 목표

- PyTorch MobileNetV2로 CIFAR-10 베이스라인 학습
- QAT 적용 후 정확도/모델 크기/추론 속도 비교
- ONNX export + 그래프 최적화
- 라즈베리파이에서 실제 배포 및 벤치마킹

## 파이프라인

```
학습 (PyTorch) → QAT fine-tuning → ONNX export → ONNX 양자화 → 엣지 배포
```

## 결과 요약

> 진행 중 — 수치는 experiments 완료 후 업데이트 예정

| 모델 | Accuracy | 모델 크기 | 추론 시간 (CPU) |
|---|---|---|---|
| MobileNetV2 (baseline) | - | - | - |
| QAT | - | - | - |
| ONNX (최적화) | - | - | - |
| ONNX (양자화) | - | - | - |

## 실행 방법

```bash
# 환경 설치
uv sync

# 베이스라인 학습
uv run python src/train.py

# ONNX 변환
uv run python src/export.py

# 벤치마킹
uv run python src/benchmark.py
```

## 환경

- Python 3.13
- PyTorch (CPU)
- onnxruntime
