Project1: ONNX기반 모델 경량화 및 엣지 다비아스 추론 성능 벤치마킹(LPCVC, MLPerfTiny 대회 참고)
핵심 주제 이미지 분류(Image Classification)
데이터셋 CIFAR-10 — MLPerf Tiny의 공식 Image Classification 벤치마크가 쓰는 것과 동일한 데이터셋. 결과를 국제 표준 벤치마크 기준(85% Top-1 정확도)이랑 비교해서 말할 수 있음. LPCVC의 NaturalScenes 데이터셋은 비공개라 못 쓰기 때문에, CIFAR-10으로 결정
모델 & 방법론
베이스 모델: MobileNetV2
참고할 방법론: HAN Lab의 Once-for-All(OFA) — 하나의 큰 네트워크를 학습시켜서 다양한 하드웨어 제약에 맞는 서브네트워크를 뽑아내는 접근. LPCVC 실제 우승 기법이라 여유 되면 이 아이디어를 응용해보면 좋음
양자화 쪽을 더 파는 것이 좋아 보임
전체 파이프라인
GPU서버에서 학습 + QAT
ONNX 변환 + 그래프 최적화
양자화(PTQ/QAT)
라즈베리파이에 배포 → 원본 vs 경량화 버전 정확도/속도/모델크기 비교

대회/벤치마크 활용 방식:
MLPerf Tiny 규격을 목표 기준선으로 사용 (CIFAR-10, 85% Top-1) — 데이터셋과 정확도 목표만 가져오고, 배포는 마이크로컨트롤러 대신 라즈베리파이로 진행
HAN Lab OFA 방법론을 기법 참고자료로 활용
(스트레치) 여유 되면 Qualcomm AI Hub로 LPCVC 스타일 원격 평가도 추가
(스트레치) 더 여유 되면 실제 마이크로컨트롤러 사서 MLPerf Tiny 공식 제출까지 도전

## 단계별 예상 소요일 (러프하게)
| 단계 | 내용 | 예상 일수 | 왜 이 정도인지 |
|---|---|---|---|
| **1. 데이터/베이스라인** | CIFAR-10 로딩, MobileNetV2 학습 (또는 전이학습) | 1~2일 | CIFAR-10은 가볍고 MobileNetV2도 흔한 조합이라 코드 자체는 빠르게 됨. GPU 서버 학습 시간이 변수 |
| **2. QAT 적용** | 오늘 배운 prepare_qat_fx 패턴 그대로 적용 + fine-tuning | 1~2일 | 오늘 이미 파이프라인을 손으로 익혔으니, 여기는 상대적으로 빠를 것 같아. 다만 CIFAR-10 학습 루프 붙이는 데 반나절 정도는 예상 |
| **3. ONNX 변환 + 그래프 최적화** | PyTorch → ONNX export, onnxruntime의 최적화 패스 적용 | 1~2일 | 오늘 FX Mode 겪었던 것처럼, ONNX export도 처음 하면 dynamic shape이나 특정 op 미지원으로 막히는 경우가 흔해서 디버깅 여지를 남겨둬야 함 |
| **4. PTQ/QAT 양자화 (ONNX 레벨)** | onnxruntime quantization API로 재적용 | 1일 | PyTorch에서 한 번 개념 잡았으니 빠르게 갈 수 있지만, "PyTorch 레벨 QAT"와 "ONNX 레벨 양자화"가 정확히 어떻게 이어지는지 정리가 필요 |
| **5. 라즈베리파이 배포 + 벤치마킹** | 실제 하드웨어에 옮겨서 원본 vs 경량화 비교 | 2~3일 | 여기가 제일 변수 커 — onnxruntime을 ARM에서 빌드/설치하는 것부터, 실제 기기 접근/환경 세팅, 크로스컴파일 이슈 등 "환경 자체"에서 막힐 가능성이 높음 |
| **6. 정리/문서화** | 결과 표, MLPerf Tiny 기준과 비교 정리 | 0.5~1일 | |