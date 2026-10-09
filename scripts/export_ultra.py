#!/usr/bin/env python3
"""
export_onnx.py of k2-fsa/sherpa-onnx (scripts/nemo/parakeet-tdt-0.6b-v3, commit
23e65586), unchanged in what it exports, with two differences: it takes the
.nemo to export as an argument, and the metadata names the fine-tune. The url
must contain "tdt": sherpa-onnx tells a TDT model from a plain transducer by
that substring (offline-transducer-nemo-model.cc, v1.13.7).
"""
import argparse
import os
from typing import Dict

import nemo.collections.asr as nemo_asr
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

URL = "https://huggingface.co/moondream/parakeet-ultra (parakeet-tdt-0.6b-v3 architecture)"


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", required=True)
    args = parser.parse_args()

    asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=args.nemo, map_location="cpu")
    asr_model.eval()

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")

    asr_model.encoder.export("encoder.onnx")
    asr_model.decoder.export("decoder.onnx")
    asr_model.joint.export("joiner.onnx")

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "model_type": "EncDecRNNTBPEModel",
        "version": "2",
        "model_author": "NeMo",
        "url": URL,
        "comment": "Only the transducer branch is exported",
        "feat_dim": 128,
    }

    for m in ["encoder", "decoder", "joiner"]:
        quantize_dynamic(
            model_input=f"./{m}.onnx",
            model_output=f"./{m}.int8.onnx",
            weight_type=QuantType.QUInt8 if m == "encoder" else QuantType.QInt8,
        )

    add_meta_data("encoder.int8.onnx", meta_data)
    os.system("ls -l *.int8.onnx tokens.txt && sha256sum *.int8.onnx tokens.txt")
    print("meta_data", meta_data)


if __name__ == "__main__":
    main()
