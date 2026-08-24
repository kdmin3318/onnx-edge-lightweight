# 실험 환경 및 소요 시간 (2026-08-18)

## 하드웨어

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA RTX A6000 (48GB) |
| CPU | Intel Core i9-10900X @ 3.70GHz (10코어 20스레드) |
| RAM | 125GB |

## 소프트웨어

| 항목 | 버전 |
|---|---|
| Python | 3.13.9 |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| CUDA (nvcc) | 11.5 / 드라이버 580.173.02 |
| onnxruntime | 1.28.0 (CPU 전용 빌드) |
| torchao | 0.18.0 |

CPU 벤치마크(latency 측정)는 실제 배포 타겟(라즈베리파이)이 GPU가 없다는 걸 감안해서, 학습은 GPU로 하되 **추론/latency 측정은 항상 CPU로 고정**해서 진행함 (`inference.py`에서 `device = torch.device("cpu")`로 고정).

## 단계별 소요 시간

| 단계 | 실행 위치 | 소요 시간 | 비고 |
|---|---|---|---|
| Baseline 학습 (20 에폭) | GPU | 에폭당 약 2분 (총 40분 내외) | CPU로 1에폭 시험 시 약 30분 — GPU가 약 15배 빠름 |
| QAT 학습 (5 에폭, torchao pt2e) | GPU | 에폭당 약 2분 40초 (총 13분 내외) | fake-quant 노드 포함이라 baseline보다 약간 느림 |
| 동적 PTQ 변환 | CPU | 수 초 (calibration 데이터 불필요) | 변환 자체는 빠르나 결과 모델의 **추론**이 매우 느림(10000장 평가에 약 9분 25초) |
| 정적 PTQ 변환 | CPU | calibration 500장 포함 수십 초 | 결과 모델 추론은 10000장 평가에 약 36초로 훨씬 빠름 |

## 참고

- 정확도는 매번 CIFAR-10 test set 전체(10,000장) 기준
- Latency는 배치=1, 워밍업 10회 제외 후 100회 평균 (자세한 방법론은 `src/inference.py`의 `measure_latency`/`measure_latency_onnx` 참고)
- 자세한 트러블슈팅 과정은 `docx/troubleshooting.md` 참고
