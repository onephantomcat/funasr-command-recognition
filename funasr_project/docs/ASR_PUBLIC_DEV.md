# ASR public-data development record

## Scope

This record uses only AISHELL-1 public data. DataSetA remains the final contest
test set and its labels are not used for ASR tuning.

## Model and runtime configuration

- ASR: `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- VAD: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- Punctuation model: disabled for command recognition to reduce memory use.
- Speaker model for optional target purification: CAM++.

## Clean ASR split

`prepare_asr_finetune_manifest.py` creates a speaker-disjoint clean manifest
from expanded AISHELL audio without copying the wav files.

| Split | Speakers | Samples |
| --- | --- | ---: |
| Train | S0002-S0014 | 4,558 |
| Dev | S0015-S0017 | 1,053 |

The first 100 dev utterances were evaluated before any fine-tuning:

| Setting | Corpus CER | Mean ASR latency |
| --- | ---: | ---: |
| Paraformer-large + VAD, punctuation disabled | 1.19% | 0.363 s/sample |

## Overlap robustness smoke test

The competition-shaped public set contains 16 target-speaker samples: four
clean, four 5 dB overlap, four 0 dB overlap, and four 5 dB babble samples.

| Setting | Corpus CER |
| --- | ---: |
| ASR only | 35.43% |
| CAM++ guided target purification + ASR | 25.56% |

The largest improvement was on the 5 dB overlap group: 44.00% to 4.00% CER.
The 0 dB overlap group did not materially improve in this small smoke test, so
purification remains opt-in rather than a default contest configuration.

## DataSetA full fair baseline

DataSetA is not a development set. The following full evaluation is the
current competition-facing baseline and does not use DataSetA labels for a
phrase bank, intent filtering, or phrase correction.

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct
```

| Metric | Result |
| --- | ---: |
| Positive / negative samples | 1,364 / 474 |
| Corpus CER | 52.87% (9,367 reference characters) |
| Positive accept rate | 69.35% |
| Rejection rate (RR) | 91.14% (432 / 474) |
| End-to-end elapsed time | 408.4 s (about 0.222 s/sample) |

This result is intentionally not comparable to the 1.19% clean AISHELL dev
CER or the 25.56% public overlap smoke test. It includes the real DataSetA
noise/domain gap and CER loss from target speech rejected by the speaker gate.
Peak memory has not yet been measured separately.

## Recommended commands

```powershell
# Rebuild clean ASR manifests after expanding AISHELL speakers.
.\.venv\Scripts\python.exe prepare_asr_finetune_manifest.py

# Measure clean speaker-disjoint dev CER.
.\.venv\Scripts\python.exe eval_asr_manifest.py `
  --manifest data\asr_finetune\aishell1_clean\dev.jsonl --limit 100

# Evaluate optional purification only for lower-similarity accepted speech.
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\public_train\aishell1 --limit 16 --sv-threshold -1 `
  --no-intent-filter --no-phrase-correct --purify --purify-sim-trigger 0.80
```

## Fine-tuning decision

The available RTX 4060 Laptop GPU has about 4 GB of VRAM. Full-parameter
fine-tuning of Paraformer-large is not recommended on this hardware. The clean
manifest is ready for a low-batch LoRA or frozen-frontend experiment, but the
current public dev baseline is already strong. Any such experiment should be
accepted only if it improves the held-out AISHELL dev CER and does not regress
the public overlap test; DataSetA is reserved for the final single evaluation.
