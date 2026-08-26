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

## 11. 비구조적(unstructured) 프루닝 — accuracy-vs-sparsity 참고 실험 (2026-08-25)

- **설정**(`src/pruning/unstructured_prune.py`): 기준은 L1 크기(작은 절댓값부터 제거), 범위는 `global_unstructured`(레이어별 개별 기준이 아니라 전체 Conv2d/Linear weight를 한 풀로 놓고 전역 순위로 제거), 대상은 Conv2d/Linear의 `weight`만(bias/BN 제외). one-shot(점진적 아님, fine-tuning 없음)으로 30/50/70/90% 네 지점 측정.
- **결과**:

  | target sparsity | accuracy |
  |---|---|
  | 30% | 94.80% |
  | 50% | 89.89% |
  | 70% | 10.64% (붕괴) |
  | 90% | 10.00% (붕괴) |

- **해석**: 50~70% 사이에 벼랑이 있음 — one-shot으로 이 구간을 넘기면 네트워크가 전혀 적응을 못 하고 랜덤 수준으로 붕괴함. 점진적 프루닝(조금씩 자르고 fine-tuning 반복)이 왜 필요한지를 실측으로 보여주는 근거.
- **배포 관점 주의**: 이 실험은 accuracy-vs-sparsity 경향만 보기 위함이고, latency/사이즈는 baseline과 사실상 동일함(onnxruntime CPU/ARM 둘 다 unstructured sparse 텐서를 활용하는 conv/matmul 커널이 없어서 0이 섞인 채로 그냥 dense 연산됨) — 그래서 `results.csv`(배포 가능 체크포인트 비교용)에는 넣지 않음. 구조적(structured) 프루닝만 실제 사이즈/latency 이득으로 이어짐(§12).
- **부가 확인**: sparse(COO) 포맷으로 저장했다고 가정했을 때의 예상 용량도 같이 계산해봄(값 4바이트 + 4차원 conv 텐서라 인덱스 4개×8바이트=32바이트, nnz당 36바이트 필요). dense(8.53MB) 대비 30%(52.93MB)/50%(37.86MB)/70%(22.78MB)는 오히려 sparse가 더 크고, 90%(7.68MB)에 가서야 dense보다 작아짐 — 계산상 손익분기점(약 88.9% = 1 - 4/36)과 실측이 정확히 일치함. "sparsity 90%는 가야 저장 용량 이득을 본다"는 통념의 근거가 바로 이 인덱스 오버헤드.

## 12. 구조적(structured) 프루닝 — `torch_pruning`, one-shot 30% 결과 (2026-08-25)

- **설정**(`src/pruning/structured_prune.py`): `torch_pruning` 1.6.1의 `MagnitudePruner` 사용. 중요도 기준은 `MagnitudeImportance(p=2)`(채널 가중치의 L2 norm), `global_pruning=True`(레이어별 균등이 아니라 전체 채널 풀에서 중요도 낮은 순), 목표 `pruning_ratio=0.3`. `classifier[1]`(출력 10클래스 고정)은 `ignored_layers`로 제외, 입력 채널 수는 dependency graph가 앞단에 맞춰 자동 조정. fine-tuning 없음(one-shot).
- **결과**: 파라미터 2,236,682 → 1,420,683 (실제 36.5% 감소, 목표 30%보다 더 잘림 — dependency graph 제약 때문). ONNX export 후 실측(`results.csv`): accuracy **12.73%**(사실상 랜덤), latency 2.05ms(baseline.onnx 1.86ms보다 오히려 소폭 증가 — 채널 수가 정돈된 숫자가 아니게 되면서 벡터화 효율이 떨어졌을 가능성), size **5.41MB**(baseline.onnx 8.77MB 대비 -38%, 파라미터 감소율과 거의 일치 — sparse 포맷/전용 커널 없이 dense 그대로 실제 용량 이득이 났다는 게 확인됨, unstructured와 대조적).
- **정확도 붕괴 원인 추정**: (1) 채널 전체 제거가 개별 가중치 제거보다 훨씬 거친 절단, (2) MobileNetV2가 depthwise separable + residual 구조라 이미 압축된 상태(capacity 여유 적음)라 프루닝 타격을 더 크게 받음, (3) `MagnitudeImportance`(가중치 크기만 봄)가 비교적 단순한 중요도 기준, (4) fine-tuning 없이 one-shot이라 회복 기회 없음. 같은 30% 근방에서도 unstructured(94.80%)와 극명하게 대조됨 — 구조적 프루닝은 같은 비율이라도 unstructured보다 훨씬 민감함을 실측으로 확인.
- **fine-tuning 결과**(`src/pruning/finetune_pruned.py`, lr=1e-4, Adam, CosineAnnealingLR, 10 epoch): **1 epoch만에 94.42%, 2 epoch에 94.75%**로 거의 baseline(95.42%)/static PTQ(94.91%) 수준까지 급격히 회복됨.
- **왜 이렇게 빨리 회복되는지**: 프루닝은 남은 가중치의 "값"을 안 건드림(채널만 삭제) — 그런데 BatchNorm의 `running_mean`/`running_var`는 잘리기 전 채널 구성 기준으로 계산된 통계가 그대로 남아있어서, 채널 삭제 직후엔 그 통계가 실제 activation 분포와 어긋나 정규화 자체가 틀어짐. 이게 20층 가까운 깊이를 거치며 누적돼서 12.73%(사실상 랜덤)까지 무너진 것 — 즉 "학습된 정보가 사라져서"가 아니라 "정규화 기준이 안 맞아서" 생긴 붕괴. fine-tuning을 시작하면 BN의 running 통계가 배치마다 지수이동평균으로 빠르게 재조정되고 가중치도 살짝만 보정되면 되므로(처음부터 재학습이 아니라 재보정), baseline 20 epoch/QAT 5 epoch보다 훨씬 빠르게(1~2 epoch) 회복됨. 프루닝 직후 급격한 정확도 붕괴 → fine-tuning 극초반 급반등은 프루닝 관련 논문에서 흔히 보고되는 패턴.
- **fine-tuning 최종(10 epoch)**: accuracy **95.13%**(baseline 95.42% 대비 -0.29점, 파라미터는 36.5% 적음). ONNX export 후 실측: latency **2.92ms** — 그런데 파라미터가 더 적은데도 `baseline.onnx`(1.86ms)보다 오히려 느림 (아래 `round_to` 항목 참고).
- **여기에 onnxruntime 정적 PTQ까지 얹어본 결과**(`pruned_structured_static.onnx`): accuracy 93.15%, latency 3.37ms, size **1.50MB**(지금까지 중 제일 작음). 근데 baseline은 PTQ만 얹었을 때 0.5점만 빠졌는데(95.42→94.91) 이 모델은 2점이나 빠짐(95.13→93.15) — 이미 채널을 36.5% 잘라내서 양자화 노이즈를 흡수할 capacity 여유가 baseline보다 적어졌기 때문으로 추정.
- **latency 이상 현상 원인 규명 및 수정**: `pruned_structured_finetuned.onnx`가 파라미터는 적은데 `baseline.onnx`보다 느린(2.92ms > 1.86ms) 원인을, `MagnitudePruner`가 채널을 자를 때 8/16의 배수 같은 정돈된 개수로 안 맞추고 중요도 순으로만 잘라서 CPU 벡터 연산(SIMD) 효율이 떨어졌기 때문으로 추정 — `pruning_ratio=0.3`에 **`round_to=8`**을 추가해서 재실행.
  - fine-tune 전(참고용, round_to만 다르게 재현): latency 2.05ms → **1.28ms**로 확 줄어듦 (accuracy는 fine-tune 전이라 9.60%로 의미 없음, 무시)
  - fine-tune 후(`pruned_structured_finetuned.onnx`, round_to=8 최종): accuracy **94.72%**, latency **1.31ms**(`baseline.onnx`의 1.86ms보다도 빠름 — 전체 체크포인트 중 최저 latency), size 5.25MB. `round_to` 가설이 실측으로 확인됨 — 구조적 프루닝에서 채널 수를 정돈된 배수로 맞추는 게 실제 속도 이득에 중요함.

## 남은 과제

- 구조적 프루닝+QAT 조합(현재는 프루닝+PTQ만 해봄), 더 높은 비율(50%+)에서 fine-tuning으로 어디까지 버티는지, 지식증류(KD) 실험 — 전부 다음 세션 과제
- 라즈베리파이 실기기에서 baseline/PTQ 재벤치마킹 (진짜 ARM int8 가속 효과 확인)
- (스트레치) ONNX+onnxruntime vs ExecuTorch 배포 스택 비교
- (future work) CIFAR-100처럼 태스크 난이도를 모델 capacity에 더 가깝게 맞춰서 PTQ/QAT 차이가 더 뚜렷해지는지 검증
