#!/usr/bin/env python3
"""The second half of export_ultra.py on its own: int8 quantization and the
metadata, for the fp32 ONNX files the first half wrote (that run was killed
for memory during the quantization: 12 GB per session on a shared box).
Without NeMo and torch in the process; add_meta_data as in export_ultra.py."""
import os
import sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

URL = "https://huggingface.co/moondream/parakeet-ultra (parakeet-tdt-0.6b-v3 architecture)"


def add_meta_data(filename, meta_data):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)

meta_data = {
    "vocab_size": 8192,
    "normalize_type": "per_feature",
    "pred_rnn_layers": 2,
    "pred_hidden": 640,
    "subsampling_factor": 8,
    "model_type": "EncDecRNNTBPEModel",
    "version": "2",
    "model_author": "NeMo",
    "url": URL,
    "comment": "Only the transducer branch is exported",
    "feat_dim": 128,
}

for m in sys.argv[1:] or ["decoder", "joiner", "encoder"]:
    quantize_dynamic(
        model_input=f"./{m}.onnx",
        model_output=f"./{m}.int8.onnx",
        weight_type=QuantType.QUInt8 if m == "encoder" else QuantType.QInt8,
    )
    print("quantized", m, flush=True)

if os.path.exists("encoder.int8.onnx"):
    add_meta_data("encoder.int8.onnx", meta_data)
os.system("ls -l *.int8.onnx tokens.txt && sha256sum *.int8.onnx tokens.txt")
