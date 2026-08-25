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

## 5. torchao `testing` 경로 `XNNPACKQuantizer`의 "부분 양자화" 진단 — 오진단으로 판명 (2026-08-25 정정)

- **최초 증상(2026-08-18)**: `convert_pt2e()` 이후 저장된 state_dict의 텐서 159개 중 106개(약 2/3)가 여전히 fp32로 남음. calibration 유무와 무관하게 동일한 결과.
- **최초 원인 추정(틀림)**: 공식 `XNNPACKQuantizer`는 `executorch` 패키지 쪽으로 이전됐고, 우리가 쓴 건 torchao의 비공식(`testing`) 경로 사본이라 MobileNetV2의 inverted residual 블록 일부를 못 커버하는 "coverage 버그"라고 판단.
- **정정(2026-08-25)**: `executorch` 공식 `XNNPACKQuantizer`로 교체해서 재실행해도 **완전히 동일한 텐서 구성(53 int8 / 106 fp32 / 159 총)**이 나옴 → coverage 버그 진단 자체가 틀렸음이 확인됨. state_dict를 직접 까보니 106개 fp32 중 53개는 원래 fp32로 남는 게 정상인 bias, 나머지 53개는 실제 연산에 안 쓰이는 잔여 fp32 사본이고, 진짜 연산은 별도의 `_frozen_paramN`(int8) 53개가 담당하고 있었음 — **53개 conv/linear 레이어는 처음부터 100% 양자화되고 있었음**.
- **9.84% collapse의 진짜 원인**: coverage 문제가 아니라 항목 9 참고.

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

## 9. `convert_pt2e()` 결과물은 ONNX로 export 불가 — `quantized_decomposed` 커스텀 op (2026-08-25)

- **증상**: `qat.pth`(`convert_pt2e()` 결과물)를 그대로 ONNX export 시도. 신형 dynamo 기반 exporter는 에러 없이 통과하지만 결과 accuracy가 9.84%(랜덤 수준)로 collapse. 구형(`dynamo=False`) exporter는 `torch.onnx.errors.UnsupportedOperatorError: ... quantized_decomposed::dequantize_per_tensor`로 명확히 실패.
- **원인**: `convert_pt2e()`가 만드는 실제 계산 그래프는 `quantized_decomposed::quantize_per_tensor`/`dequantize_per_tensor`라는 torchao/executorch 전용 커스텀 op를 씀(표준 ATen op 아님). ONNX exporter(신구 버전 모두)가 이 op의 의미를 모름 — 구버전은 명확히 에러로 막고, 신버전은 억지로 변환하다 그래프가 의미상 깨져서 9.84% collapse로 이어짐. 항목 5의 "부분 양자화"가 아니라 이게 진짜 원인이었음.
- **결론**: pt2e의 `convert_pt2e()` 결과물은 ONNX 배포 경로와 근본적으로 안 맞음 — executorch 자체 런타임(`to_edge_transform_and_lower` → `.pte`)으로 lowering해야 쓸 수 있는 표현임. ONNX 배포를 유지하는 한 이 결과물을 직접 export하는 시도는 하지 않기로 함.

## 10. 해결책: QAT는 "더 나은 fp32 시작점"으로만 쓰고, 실제 int8 변환은 onnxruntime에 위임 (2026-08-25)

- **아이디어**: `prepare_qat_pt2e` 단계에서 이미 conv+BN이 하나로 folding됨(state_dict에 `<conv>.weight`/`<conv>.weight_bias` 형태로 저장). 이 접힌 fp32 가중치만 꺼내서 별도 스크립트(`qat_fp32_export.py`)로 plain MobileNetV2 구조(conv `bias=True` + BN 자리를 `Identity`로 대체)에 재배치 → 커스텀 op 없이 기존 `export.py`(fp32→ONNX) → `ptq_static.py`(onnxruntime 정적 PTQ) 파이프라인 그대로 재사용.
- **결과**: `qat_fp32.onnx`(순수 fp32, 양자화 노이즈 없음) 93.33% / `qat_static.onnx`(최종 int8) 94.50% — pt2e 자체의 "real int8" 기록(94.57%, 학습 중 전체 데이터로 계산한 calibration 기준)과 거의 동급.
- **부가 발견(재현성 이슈)**: pt2e에서 activation의 scale/zero_point는 state_dict에 저장되지 않고, `convert_pt2e()` 호출 직전에 흘린 calibration 데이터를 기준으로 그래프에 상수로 박힘. 그래서 동일한 가중치를 다시 불러와도 calibration 데이터가 다르면 정확도가 달라짐 — 직접 재현 실험(64장짜리 즉석 calibration)에서 93.44%가 나와, 학습 중 기록된 94.57%(전체 데이터 기준)와 차이가 남을 확인. `qat.pth`의 int8 정확도 수치는 항상 "언제/얼마나 calibration 했는지"를 같이 밝혀야 함.
- **최종 평가**: baseline PTQ(94.91%)와 비교해 QAT+PTQ(94.50%)가 오히려 근소하게 낮음. 원인 추정: (1) QAT가 사실상 1 epoch만 유효하게 학습됨(lr=1e-5로 5 epoch 중 1 epoch째에서만 best 갱신), (2) baseline 자체가 이미 PTQ만으로도 손실이 작음(0.5%p) — CIFAR-10 10클래스 대비 ImageNet 사전학습 MobileNetV2의 capacity가 과함(태스크 난이도 대비 여유 capacity가 커서 양자화 노이즈를 쉽게 흡수). 이번 세팅에서는 QAT의 이득이 뚜렷하게 드러나지 않았다는 것 자체가 결과.

## 남은 과제

- 라즈베리파이 실기기에서 baseline/PTQ 재벤치마킹 (진짜 ARM int8 가속 효과 확인)
- (스트레치) ONNX+onnxruntime vs ExecuTorch 배포 스택 비교
- (future work) CIFAR-100처럼 태스크 난이도를 모델 capacity에 더 가깝게 맞춰서 PTQ/QAT 차이가 더 뚜렷해지는지 검증
