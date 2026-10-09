# Parakeet Ultra for sherpa-onnx (int8)

[Moondream's Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra), a post-trained
[NVIDIA parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), converted to the
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) `nemo_transducer` format and quantized to int8.
The Plotdesk desktop app downloads these files to transcribe meetings on the device: 25 European
languages, no audio leaves the computer.

Format conversion and quantization only, no training.

## Files

Release [`v1`](https://github.com/plotdesk/parakeet-ultra-sherpa-onnx/releases/tag/v1):

| File | Bytes | SHA-256 |
|---|---|---|
| `encoder.int8.onnx` | 652282328 | `21e533084732d6c13892744f31e5d5305c74330b6f8c923a75ba3b3500325694` |
| `decoder.int8.onnx` | 11845274 | `1fab98fe6c12aded87d2da66272cc9e148d0a0044ce3850a12fe56302ec4a922` |
| `joiner.int8.onnx` | 6355277 | `8a71aaccfdba3d451775507a889afd2257d2e754463fd81095a8c8f290f924c2` |
| `tokens.txt` | 93939 | `d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d` |

The release also carries `SHA256SUMS`.

## License and attribution

- Base model: **parakeet-tdt-0.6b-v3** by NVIDIA, revision `541d1f99`, licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Fine-tune: **Parakeet Ultra** by Moondream, revision `73175eb7`, licensed CC BY 4.0.
- These files: converted and quantized by Plotdesk, licensed CC BY 4.0. Changes: weights mapped from the
  Hugging Face layout to NeMo by renaming keys, the six `vad_head.*` tensors of Parakeet Ultra dropped
  (not part of the NeMo architecture), exported to ONNX and quantized to int8.

## How the files were made

The scripts are in [`scripts/`](scripts/). Versions: Python 3.12.3, nemo_toolkit 3.0.0, torch 2.14.1+cpu,
onnx 1.23.2, onnxruntime 1.31.0.

1. **Inputs**, checked against the SHA-256 Hugging Face lists for them:
   - `nvidia/parakeet-tdt-0.6b-v3` @ `541d1f99`: `parakeet-tdt-0.6b-v3.nemo`
     (`3cbdc858…d10d`) and `model.safetensors` (`3a202636…5e11`)
   - `moondream/parakeet-ultra` @ `73175eb7`: `model.safetensors` (`c9608f36…17dc`)
2. **Weights onto NeMo** (`scripts/hf_to_nemo.py`): for every tensor of the NeMo model of v3 the Hugging
   Face name is computed with the mapping of `transformers`
   ([`convert_nemo_to_hf.py`](https://github.com/huggingface/transformers/blob/f36c750069f0aef45179207a680b592f138450ec/src/transformers/models/parakeet/convert_nemo_to_hf.py)
   @ `f36c7500`, placed next to the script) and its tensor taken, cast from F16 to F32. The featurizer's
   window and filterbank keep the values of v3.
   *Check:* the same mapping applied to NVIDIA's own `model.safetensors` reproduces all 723 tensors of the
   `.nemo` bit for bit.
3. **Export** (`scripts/export_ultra.py`): the `export_onnx.py` of
   [k2-fsa/sherpa-onnx `scripts/nemo/parakeet-tdt-0.6b-v3`](https://github.com/k2-fsa/sherpa-onnx/tree/23e65586527871859acfab73048a8bb83bbae856/scripts/nemo/parakeet-tdt-0.6b-v3)
   @ `23e65586`, unchanged in what it exports. The metadata `url` names Parakeet Ultra and contains
   `tdt`: sherpa-onnx tells a TDT model by that substring and decodes it as a plain transducer otherwise.
4. **int8** with onnxruntime's dynamic quantization, as in that script (`QUInt8` for the encoder, `QInt8`
   for decoder and joiner). On a box with 12 GB the run of step 3 was killed while quantizing the encoder;
   `scripts/quantize_ultra.py` did this step on its own with the same parameters (peak 6.6 GB).

## Checks

- **Against an independent conversion**
  ([tpmaerk/parakeet-ultra-sherpa-onnx-int8](https://huggingface.co/tpmaerk/parakeet-ultra-sherpa-onnx-int8),
  made the same way from the same revisions): the same graphs and inputs. `joiner.int8.onnx` and
  `tokens.txt` are identical byte for byte. In the encoder, 31 folded constants differ by at most
  1e-6 (torch version) and `model_author` reads `NeMo`. In the decoder, the newer onnxruntime quantizes
  the embedding asymmetrically (scale 0.0388, zero point 124 instead of 0.0399 and 128); both
  dequantize to within half a step of Parakeet Ultra's weights.
- **Recognition**: see the measurement below.

## Measurement

FLEURS German test split, every third clip (288 clips, 63.5 minutes), sherpa-onnx-node 1.13.7 on Linux
x86-64 (8 threads on a shared 16-core box), greedy search, the recognizer config of the Plotdesk desktop app.
Words lower-cased and split at everything that is not a letter or a digit; numbers are not normalized.

| Model | WER | Real-time factor |
|---|---|---|
| Parakeet Ultra int8 (this release) | 5.45 % | 8.8× |
| Whisper large-v3 turbo int8 (k2-fsa), language `de` | 8.29 % | 2.4× |

The same 288 clips, the same box, one after the other. Through the desktop app's whole meeting path
(voice detection, diarization, pieces of at most 28 s) a German test meeting of 3:46 min built from 16
FLEURS clips came out at 6.54 % (Parakeet Ultra) against 7.59 % (Whisper turbo); recognition took
18 s against 97 s.

## Usage (sherpa-onnx, Node)

```js
const sherpa = require('sherpa-onnx-node');
const recognizer = new sherpa.OfflineRecognizer({
  featConfig: { sampleRate: 16000, featureDim: 128 },
  modelConfig: {
    transducer: { encoder: 'encoder.int8.onnx', decoder: 'decoder.int8.onnx', joiner: 'joiner.int8.onnx' },
    tokens: 'tokens.txt',
    modelType: 'nemo_transducer',
    numThreads: 4,
  },
  decodingMethod: 'greedy_search',
});
```

Keep pieces short: cut at pauses, at most about 30 s. In one go the quality drops with length.
