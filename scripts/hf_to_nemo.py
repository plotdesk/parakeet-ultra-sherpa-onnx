#!/usr/bin/env python3
"""
Puts the weights of a Hugging Face Parakeet TDT checkpoint (model.safetensors,
transformers' ParakeetForTDT layout) onto the NeMo model of
nvidia/parakeet-tdt-0.6b-v3 and saves that as a .nemo.

Key renaming only, through the mapping transformers uses the other way round
(convert_nemo_to_hf.py of transformers @ f36c7500, placed next to this
script): for every NeMo key the HF name is computed and its tensor taken. NeMo keys without an HF counterpart (the featurizer's window and
filterbank) keep the v3 values, HF keys without a NeMo counterpart are listed.

  hf_to_nemo.py --nemo v3.nemo --hf ultra.safetensors --out ultra.nemo
  hf_to_nemo.py --nemo v3.nemo --hf v3.safetensors --check   # must be bit-exact
"""
import argparse
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from convert_nemo_to_hf import NEMO_TDT_WEIGHT_MAPPING, NEMO_TO_HF_WEIGHT_MAPPING, convert_key  # noqa: E402

MAPPING = {**NEMO_TO_HF_WEIGHT_MAPPING, **NEMO_TDT_WEIGHT_MAPPING}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--out")
    parser.add_argument("--check", action="store_true", help="compare instead of save")
    args = parser.parse_args()

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cpu")
    nemo_state = model.state_dict()
    hf_state = load_file(args.hf)

    new_state, kept, used = {}, [], set()
    for key, value in nemo_state.items():
        hf_key = convert_key(key, MAPPING)
        if hf_key not in hf_state:
            kept.append(key)
            new_state[key] = value
            continue
        tensor = hf_state[hf_key]
        if tuple(tensor.shape) != tuple(value.shape):
            raise SystemExit(f"shape mismatch {key} {tuple(value.shape)} <- {hf_key} {tuple(tensor.shape)}")
        new_state[key] = tensor.to(value.dtype)
        used.add(hf_key)

    unused = sorted(set(hf_state) - used)
    dtypes = sorted({str(t.dtype) for t in hf_state.values()})
    print(f"NeMo tensors: {len(nemo_state)}, taken from HF: {len(used)}, kept from v3: {kept}")
    print(f"HF tensors not used: {unused}")
    print(f"HF dtypes: {dtypes}")

    if args.check:
        differ = [k for k in nemo_state if k not in kept and not torch.equal(nemo_state[k], new_state[k])]
        print(f"bit-exact: {len(differ) == 0} ({len(differ)} differ)")
        sys.exit(1 if differ else 0)

    model.load_state_dict(new_state, strict=True)
    model.save_to(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
