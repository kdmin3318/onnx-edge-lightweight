import argparse
import json
import os
from collections import defaultdict

import numpy as np
import onnxruntime as ort

# onnxruntime 내장 프로파일러로 op별 실행 시간 breakdown 확인.
# static PTQ가 fp32 ONNX보다 2배 느린 이유가 (a) QuantizeLinear/DequantizeLinear
# 자체의 고정 오버헤드 때문인지, (b) 이 x86 CPU의 int8 conv 커널이 fp32 conv만큼
# 최적화가 안 된 건지 구분하기 위함 (onnxruntime 내부 소스는 안 건드림, 제공되는
# 프로파일링 옵션만 사용).


def profile_model(checkpoint, runs=50):
    so = ort.SessionOptions()
    so.enable_profiling = True
    session = ort.InferenceSession(checkpoint, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    for _ in range(10):
        session.run(None, {input_name: x})
    for _ in range(runs):
        session.run(None, {input_name: x})

    return session.end_profiling()


def summarize(profile_path, top_n=15):
    with open(profile_path) as f:
        events = json.load(f)

    by_optype = defaultdict(float)
    total = 0.0
    for e in events:
        if e.get("cat") != "Node":
            continue
        dur = e.get("dur", 0) / 1000.0  # us -> ms
        op_type = e.get("args", {}).get("op_name", e.get("name", "?"))
        by_optype[op_type] += dur
        total += dur

    print(f"\n=== {profile_path} ===")
    print(f"Total node time: {total:.2f} ms")
    for op_type, dur in sorted(by_optype.items(), key=lambda x: -x[1])[:top_n]:
        print(f"  {op_type:30s} {dur:10.2f} ms  ({100*dur/total:5.1f}%)")

    os.remove(profile_path)  # 임시 산출물이라 분석 후 바로 정리


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/baseline_static.onnx")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    profile_path = profile_model(args.checkpoint, args.runs)
    summarize(profile_path)
