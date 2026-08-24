import argparse

from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from common import DATA_DIR, test_transform

# onnxruntime의 정적 PTQ. calibration 데이터를 미리 흘려서 activation의 scale/zero_point를
# 고정값으로 계산해두고, 그 값을 그래프에 상수로 박아넣음. CNN엔 이게 표준 방식.
CALIBRATION_SIZE = 500  # 전체 10000장 다 안 돌려도 충분히 대표성 있는 범위 확보 가능


class CIFAR10CalibrationReader(CalibrationDataReader):
    def __init__(self, input_name):
        dataset = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)
        subset = Subset(dataset, range(CALIBRATION_SIZE))
        loader = DataLoader(subset, batch_size=1, shuffle=False)
        self.iterator = iter(loader)
        self.input_name = input_name

    def get_next(self):
        batch = next(self.iterator, None)
        if batch is None:
            return None
        images, _ = batch
        return {self.input_name: images.numpy()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/baseline.onnx")
    parser.add_argument("--output", default="checkpoints/baseline_static.onnx")
    args = parser.parse_args()

    calibration_reader = CIFAR10CalibrationReader(input_name="input")

    quantize_static(
        model_input=args.checkpoint,
        model_output=args.output,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        use_external_data_format=False,
    )

    print(f"Exported: {args.output}")
