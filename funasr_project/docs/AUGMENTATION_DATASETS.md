# MUSAN and RIRS_NOISES Augmentation

## Purpose

MUSAN and RIRS_NOISES are external augmentation assets. They do not supply
competition labels and are never mixed with DataSetA. Existing AISHELL target
speech remains the transcript source; MUSAN adds noise/background speech and
RIRS_NOISES simulates far-field room response.

## Download

```powershell
cd C:\Users\13238\Desktop\挑战杯1号语音识别\新\funasr_project
.\.venv\Scripts\python.exe prepare_augmentation_assets.py `
  --assets musan,rirs_noises --source official
```

The downloader stores resumable `.part` files and extracts assets to:

```text
data/public/augmentations/
  musan/extracted/
  rirs_noises/extracted/
```

Use `--proxy http://127.0.0.1:7890` when a local proxy is required. The China
mirror is tried by `--source auto`, but the official source is recommended when
the mirror certificate is unavailable.

## Build Competition-Shaped External Data

After AISHELL and both augmentation assets are ready, run:

```powershell
.\.venv\Scripts\python.exe build_external_trainset.py `
  --wav-root data\public\aishell1\extracted\data_aishell\wav_expanded\train `
  --csv data\public\aishell1\extracted\data_aishell\transcript\aishell_transcript_v0.8.txt `
  --out data\public_train\aishell1_musan_rirs `
  --noise-root data\public\augmentations\musan\extracted `
  --rir-root data\public\augmentations\rirs_noises\extracted `
  --target-speakers 8 --interferer-speakers 8 --trials-per-speaker 4 `
  --noise-snr-db 5 --reverb-noise-snr-db 5 --seed 2026
```

The resulting `pos.jsonl` contains clean, two-speaker overlap, multi-speaker
babble, MUSAN noise, and RIR plus noise target-speech samples. `neg.jsonl`
contains other-speaker command audio with an empty `识别文本` label. Every row
retains the contest input fields: `id`, `唤醒音频`, `唤醒文本`, `识别音频`, and
`识别文本`.

## Guardrails

- Build phrase banks, gate thresholds, and augmentation policies only from this
  external data.
- Do not use DataSetA labels, its `pos/neg` directory names, or its audio for
  training or parameter selection.
- Keep the seed, source archive URL, and generated manifest path in the
  experiment record.
