import torch
import torch.nn as nn

from common import build_model
from export import export_to_onnx, check_onnx_model, verify_parity

QAT_CHECKPOINT = "checkpoints/qat.pth"
OUTPUT_PATH = "checkpoints/qat_fp32.onnx"

# qat.pth는 prepare_qat_pt2e 단계에서 각 conv 뒤의 BatchNorm이 conv weight/bias에
# 수학적으로 접혀들어간(folded) 상태로 저장돼있음 — eval 모드에서 BN(conv(x))와
# 완전히 동일한 값을 내는 conv'(x) 하나로 합쳐진 형태(이름이 "<conv>.weight_bias"인
# 키들). 그래서 원래 conv(bias=False)+BN 구조로 그대로 못 불러오고, conv에 bias를
# 켜고 바로 뒤따르는 BN을 항등변환(Identity)으로 비운 다음 접힌 weight/bias를
# 그대로 꽂아넣어야 함. convert_pt2e()가 만드는 quantized_decomposed 그래프는
# 아예 안 건드리므로(state_dict의 순수 fp32 텐서만 사용) custom op가 생길 일이 없음.


def get_by_path(root, path):
    obj = root
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def set_by_path(root, path, new_module):
    *parents, last = path.split(".")
    obj = root
    for p in parents:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    if last.isdigit():
        obj[int(last)] = new_module
    else:
        setattr(obj, last, new_module)


def fold_bn_and_load(model, state_dict):
    # "<conv_path>.weight_bias" 키가 있는 곳 = BN이 접혀 들어간 conv.
    # bn은 항상 같은 Sequential 안에서 conv 바로 다음 인덱스에 위치함.
    weight_bias_keys = [k for k in state_dict if k.endswith(".weight_bias")]

    for wb_key in weight_bias_keys:
        conv_path = wb_key[: -len(".weight_bias")]
        parts = conv_path.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        bn_path = ".".join(parts)

        old_conv = get_by_path(model, conv_path)
        new_conv = nn.Conv2d(
            old_conv.in_channels, old_conv.out_channels, old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding,
            dilation=old_conv.dilation, groups=old_conv.groups, bias=True,
        )
        new_conv.weight = nn.Parameter(state_dict[conv_path + ".weight"].clone())
        new_conv.bias = nn.Parameter(state_dict[wb_key].clone())

        set_by_path(model, conv_path, new_conv)
        set_by_path(model, bn_path, nn.Identity())

    # classifier.1은 BN이 없는 순수 Linear라 접힘 없이 그대로 로드
    model.classifier[1].weight = nn.Parameter(state_dict["classifier.1.weight"].clone())
    model.classifier[1].bias = nn.Parameter(state_dict["classifier.1.bias"].clone())

    return model


if __name__ == "__main__":
    state_dict = torch.load(QAT_CHECKPOINT, map_location="cpu", weights_only=False)

    model = build_model(pretrained_backbone=False)
    model = fold_bn_and_load(model, state_dict)
    model.eval()

    export_to_onnx(model, OUTPUT_PATH)
    check_onnx_model(OUTPUT_PATH)
    is_close, max_diff = verify_parity(model, OUTPUT_PATH)

    print(f"Exported: {OUTPUT_PATH}")
    print("ONNX graph check: OK")
    print(f"PyTorch vs ONNX output match: {'OK' if is_close else 'MISMATCH'} (max diff: {max_diff:.2e})")
