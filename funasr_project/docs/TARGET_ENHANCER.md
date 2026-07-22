# CER-Oriented Target Speech Enhancer

## Purpose

`target_enhancer.py` is a lightweight, trainable waveform front end placed
before CAM++ verification and Paraformer recognition. It estimates a
time-frequency mask from the command mixture and a same-speaker wake audio
summary. Training targets are the clean AISHELL-1 target utterances; inputs
are synthetically mixed only from AISHELL-1, MUSAN, and RIRS_NOISES.

This is distinct from `train_lightweight_gate.py`: the gate changes the
accept/reject decision and mainly affects RR, whereas the enhancer changes the
audio fed to Paraformer and is therefore the CER-focused experiment.

## Training

Wait for MUSAN extraction to complete, then run from `funasr_project`:

```powershell
..\.venv\Scripts\python.exe train_target_enhancer.py `
  --wav-root data\public\aishell1\extracted\data_aishell\wav_expanded\train `
  --noise-root data\public\augmentations\musan\extracted `
  --rir-root data\public\augmentations\rirs_noises\extracted `
  --out models\target_enhancer_musan_rirs.pt `
  --epochs 8 --batch-size 6 --steps-per-epoch 250 --dev-steps 40
```

The training/validation speakers are disjoint. Do not use DataSetA files,
labels, or paths in this command. The output checkpoint and adjacent JSON file
record the source data, held-out speaker IDs, and best validation loss.

## DatasetA CER Comparison

Keep the same fair decision settings for both runs. The only experimental
difference is the learned external-data checkpoint:

```powershell
# Frozen baseline.
..\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct `
  --out data\datasetA\eval_report_cer_baseline.json

# CER front-end experiment.
..\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct `
  --enhancer-model models\target_enhancer_musan_rirs.pt `
  --enhancer-dir data\datasetA\enhanced_cache `
  --out data\datasetA\eval_report_cer_enhancer.json
```

Compare `positive_corpus_cer` first, then confirm that `positive_accept_rate`
does not collapse and record the RR and latency trade-offs. The enhancer is
adopted only when it improves the held-out external development loss and gives
a reproducible DataSetA CER gain under the same evaluation configuration.
