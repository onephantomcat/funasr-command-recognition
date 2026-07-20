# ASR public-data development record

## Scope

This record uses only AISHELL-1 public data. DataSetA is retained for stage
evaluation, while the organizer's test set B is used for the final ranking;
neither set's labels are used for ASR tuning.

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

## DataSetA full stage baseline

DataSetA is not a training set. The following full evaluation is a stage
baseline and does not use DataSetA labels for a phrase bank, intent filtering,
or phrase correction. It is not a final test set B result.

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct
```

| Metric | Result |
| --- | ---: |
| Positive / negative samples | 1,364 / 474 |
| Local raw-character CER | 53.43% (9,515 reference characters) |
| Positive accept rate | 69.35% |
| Rejection rate (RR) | 91.14% (432 / 474) |
| End-to-end elapsed time | 438.9 s (about 0.239 s/sample) |
| Local process working set | 3,073.22 MB |

This result is intentionally not comparable to the 1.19% clean AISHELL dev
CER or the 25.56% public overlap smoke test. It includes the real DataSetA
noise/domain gap and CER loss from target speech rejected by the speaker gate.
The v0.3.4 result removes whitespace inserted by Paraformer between Chinese
characters before emitting the hypothesis; it does not use labels, phrase
correction, or a command vocabulary. The v0.3.3 119.17% value includes those
formatting spaces as raw CER insertions. The organizer scorer on test set B
remains authoritative. `eval_datasetA.py` reports local process working-set
and CUDA peak-allocation diagnostics; the organizer's uniform-hardware memory
measurement remains the official one.

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
the public overlap test; DataSetA is reserved for stage reporting and test set
B is reserved for the final organizer evaluation.
