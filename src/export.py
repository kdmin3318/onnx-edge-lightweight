import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import load_model


def export_to_onnx(model, output_path, input_size=(1, 3, 224, 224)):
    dummy_input = torch.randn(input_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        # 새 dynamo 기반 exporter(기본값)가 만드는 그래프는 shape 정보가 불완전해서
        # onnxruntime 양자화 전처리(quant_pre_process)가 실패함. 예전 TorchScript 기반
        # exporter가 더 표준적인 그래프를 만들어서 이걸로 고정함.
        dynamo=False,
    )


def check_onnx_model(output_path):
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


def verify_parity(model, output_path, input_size=(1, 3, 224, 224), atol=1e-4):
    dummy_input = torch.randn(input_size)
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]

    max_diff = np.abs(torch_output - onnx_output).max()
    is_close = np.allclose(torch_output, onnx_output, atol=atol)
    return is_close, max_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/baseline.pth")
    parser.add_argument("--output", default="checkpoints/baseline.onnx")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = load_model(args.checkpoint, device)

    export_to_onnx(model, args.output)
    check_onnx_model(args.output)
    is_close, max_diff = verify_parity(model, args.output)

    print(f"Exported: {args.output}")
    print("ONNX graph check: OK")
    print(f"PyTorch vs ONNX output match: {'OK' if is_close else 'MISMATCH'} (max diff: {max_diff:.2e})")
