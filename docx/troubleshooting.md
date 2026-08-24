# 양자화 파이프라인 트러블슈팅 기록 (2026-08-18)

오늘 baseline → QAT → PTQ(동적/정적) 순으로 양자화를 진행하면서 겪은 기술적 문제들과 원인, 해결/결정 사항을 시간 순으로 정리.

## 1. `torch.ao.quantization.quantize_fx`(FX Graph Mode) collapse 버그

- **증상**: QAT 학습(fake-quant 상태)은 94%대 정확도가 잘 나오는데, `convert_fx()`로 실제 int8 변환하자마자 정확도가 10%(=랜덤 수준)로 붕괴. 모든 이미지를 클래스 하나로만 예측.
- **원인**: PyTorch가 2.10부터 `torch.ao.quantization`(FX/eager 방식 통합) 자체를 삭제 수순에 들어감. 지금 서버 torch 버전(2.13)은 이미 그 이후라 `quantize_fx` 경로가 사실상 방치된 상태.
- **해결**: 공식 후속 경로인 **torchao의 pt2e API**(`prepare_qat_pt2e`/`convert_pt2e`, `torch.export` 기반)로 마이그레이션.

## 2. `torch.export`의 배치 크기 고정 문제

- **증상**: 예시 입력 배치를 1로 주고 `Dim`으로 동적 처리 시도했더니 "코드가 배치를 상수 1로 특수화했다"는 에러.
- **원인**: PyTorch는 크기가 1인 차원을 브로드캐스팅 때문에 무조건 고정값으로 취급함(잘 알려진 함정).
- **해결**: 예시 입력 배치를 1이 아닌 값(BATCH_SIZE)으로 주고 `Dim(min=1, max=...)`으로 동적 처리 → 배치 1/16/64 전부 하나의 그래프로 처리 가능해짐.

## 3. `torch.load`의 `weights_only` 기본값 변경 (PyTorch 2.6+)

- **증상**: 양자화 모델(FX GraphModule 등 특수 구조 포함) 로딩 시 `UnpicklingError`.
- **원인**: PyTorch 2.6부터 `torch.load`의 `weights_only` 기본값이 `True`로 바뀌어 순수 텐서 외의 객체 복원이 기본 차단됨(보안 강화).
- **해결**: 우리가 직접 만든 신뢰 가능한 체크포인트이므로 `weights_only=False` 명시.

## 4. FX GraphModule + 양자화 조합의 `torch.save` 재구성 버그

- **증상**: `torch.save(quantized_model, path)`로 저장한 모델을 다시 불러오면 `AttributeError: 'Conv2d' object has no attribute '_modules'`.
- **원인**: 모델 전체 저장은 로딩 시 저장해둔 forward 소스코드를 재실행해서 그래프를 다시 추적(retrace)하는 방식인데, 이 재구성 과정이 양자화된 레이어를 제대로 못 다룸.
- **해결**: (FX 경로에서는) TorchScript(`torch.jit.script`/`torch.jit.save`)로 우회 → 이후 pt2e로 전환하면서 이 이슈 자체가 회피됨.

## 5. torchao `testing` 경로 `XNNPACKQuantizer`의 부분 양자화 문제 (미해결, QAT 결과 신뢰 불가)

- **증상**: `convert_pt2e()` 이후 저장된 state_dict의 텐서 159개 중 106개(약 2/3)가 여전히 fp32로 남음. calibration 유무와 무관하게 동일한 결과.
- **원인 추정**: 공식 `XNNPACKQuantizer`는 `executorch` 패키지 쪽으로 이전됐고, 우리가 쓴 건 torchao의 **비공식(`testing`) 경로 사본**. MobileNetV2의 inverted residual 블록 일부 패턴을 이 사본이 완전히 커버하지 못하는 것으로 추정(conv2d 노드 전체에 annotation은 붙지만 실제 변환은 절반만 이뤄짐).
- **연쇄 문제**: 이 절반짜리 양자화 모델을 ONNX로 export했더니 accuracy가 또 9.84%로 collapse.
- **결정**: `executorch` 설치(무거운 의존성, 배포 계획인 ONNX와 무관)까지는 오늘 진행하지 않고, **QAT의 정량적 결과(정확도/latency/사이즈)는 신뢰 불가로 보류**. 트러블슈팅 기록으로만 남기고 PTQ(onnxruntime, 성숙한 별도 구현)로 오늘의 핵심 수치를 확보하는 쪽으로 방향 전환.

## 6. `torch.onnx.export`(dynamo 기반, 기본값)의 불완전한 shape 정보

- **증상**: `baseline.onnx`에 onnxruntime 동적 PTQ(`quantize_dynamic`) 적용 시 `ShapeInferenceError: Inferred shape and existing shape differ`, 이후 전처리(`quant_pre_process`)로 우회 시도했으나 그마저 `Incomplete symbolic shape inference`로 실패.
- **원인**: PyTorch 2.9부터 기본값이 된 새 dynamo 기반 exporter가 만든 그래프의 shape 메타데이터가 onnxruntime의 shape inference 도구와 완전히 호환되지 않음.
- **해결**: `torch.onnx.export(..., dynamo=False)`로 예전 TorchScript 기반 exporter를 명시적으로 사용 → 문제 자체가 사라짐 (전처리 단계도 불필요해짐).

## 7. 프로젝트 루트에 랜덤 이름 부산물 파일 생성 (`sym_shape_infer_temp.onnx`, `<uuid>.data`)

- **증상**: 양자화 스크립트 실행 후 프로젝트 루트에 의도치 않은 파일들이 생성됨.
- **원인**: ONNX 라이브러리가 가중치를 외부(`.data`) 파일로 분리 저장할 때, 저장 위치(`location`)를 명시하지 않으면 **랜덤 UUID 파일명**을 기본값으로 사용함(`onnx/external_data_helper.py`). `quantize_dynamic()`이 입력 모델(`baseline.onnx`)의 external data 여부를 보고 자동으로 이 모드를 켜버림.
- **해결**: 결과물이 2GB급이 아니라 애초에 external data가 필요 없으므로 `use_external_data_format=False`로 명시 → 부산물 생성 자체가 사라짐.

## 8. 동적 PTQ vs 정적 PTQ — CNN에서의 실측 비교

동적 양자화는 PyTorch에서 Conv2d를 지원하지 않지만, onnxruntime은 지원함(`QLinearConv`). 그래서 "정말 CNN엔 정적이 나은가"를 실제로 비교해봄.

| 모델 | 정확도 | Latency (batch=1) | 사이즈 |
|---|---|---|---|
| baseline.pth (fp32) | 95.42% | 10.33ms | 8.77MB |
| baseline.onnx (fp32) | 95.42% | 1.86ms | 8.77MB |
| baseline_dynamic.onnx (동적 PTQ) | 94.99% | 82.65ms | 2.30MB |
| baseline_static.onnx (정적 PTQ) | 94.91% | 4.04ms | 2.29MB |

- 동적 PTQ는 정확도/사이즈는 괜찮지만 **fp32 ONNX보다 44배 느림** — CNN은 레이어(주로 Conv)마다 activation 범위를 실시간 계산하는 오버헤드가 커서, onnxruntime이 지원은 해도 실질적으론 안 맞는다는 게 수치로 확인됨.
- 정적 PTQ는 동적 대비 20배 이상 빠르고 사이즈도 비슷하게 줄어 사실상 CNN엔 정적이 압도적으로 유리함을 확인.
- 다만 정적 PTQ도 **fp32 ONNX(1.86ms)보다는 2배 느림** — x86 서버의 onnxruntime이 int8 커널을 fp32만큼 최적화하지 못하는 것으로 보임. 진짜 int8 가속(ARM SIMD/XNNPACK)은 실제 타겟인 라즈베리파이에서 재측정해야 확인 가능 → 다음 단계 과제.

## 남은 과제

- QAT의 정확한 정량화(전체 레이어 양자화 커버리지 확보) — 시간 되면 `executorch` 도입 검토
- 라즈베리파이 실기기에서 baseline/PTQ 재벤치마킹 (진짜 ARM int8 가속 효과 확인)
- (스트레치) ONNX+onnxruntime vs ExecuTorch 배포 스택 비교
