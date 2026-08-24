import argparse

from onnxruntime.quantization import quantize_dynamic, QuantType

# onnxruntime의 동적 PTQ. calibration 데이터 없이, activation의 scale/zero_point를
# 추론 시점에 그래프 안(DynamicQuantizeLinear)에서 계산하도록 변환함.
# CNN(Conv 위주)엔 정적 양자화가 정석이지만, "왜 정적이 CNN에 더 나은지"를 우리
# 데이터로 직접 비교해보기 위해 동적도 만들어봄.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/baseline.onnx")
    parser.add_argument("--output", default="checkpoints/baseline_dynamic.onnx")
    args = parser.parse_args()

    quantize_dynamic(
        model_input=args.checkpoint,
        model_output=args.output,
        weight_type=QuantType.QInt8,
        # 결과물이 작아서(2GB급 아님) external data로 쪼갤 필요 없음. 안 끄면 랜덤
        # UUID 파일명으로 프로젝트 루트에 부산물이 생김.
        use_external_data_format=False,
    )

    print(f"Exported: {args.output}")
