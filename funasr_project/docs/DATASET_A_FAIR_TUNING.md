# DataSetA Fair Tuning Notes

DataSetA is a stage-development and temporary-leaderboard set. Test set B is
the final script-validation and ranking set:

- `pos` evaluates target-speaker character error rate (CER).
- `neg` evaluates non-target rejection rate (RR).
- The `识别文本` / `璇嗗埆鏂囨湰` field is label-only.
- Wake audio, wake text, and recognition audio are model inputs.

Therefore, do not train, build a phrase bank, or correct ASR output from
DataSetA labels. Do not use the `pos/neg` directory split as an inference
feature: test set B may mix both types and will not expose this split.

## External Train/Tuning Set

Preferred flow: download a public dataset, then convert it to the competition
format. AISHELL-1 from OpenSLR SLR33 is currently implemented. The full archive
is large (about 15 GB), so the command supports resume and can reuse existing
files. By default the downloader uses a China-friendly OpenSLR mirror
(`openslr.magicdatatech.com`) and shows progress, speed, and ETA:

```powershell
cd C:\Users\13238\Desktop\挑战杯1号语音识别\新\funasr_project
.\.venv\Scripts\python.exe prepare_public_dataset.py --dataset aishell1 --out data\public_train\aishell1
```

If the mirror is still slow and a local proxy/VPN is listening on port 7890, run:

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py --dataset aishell1 --out data\public_train\aishell1 --source auto --proxy http://127.0.0.1:7890
```

For a quick local verification using the already-present AISHELL subset:

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py --use-existing-local --out data\public_train\aishell1_local
```

The older direct builder is still available when wav/transcript paths are
already known:

```powershell
cd C:\Users\13238\Desktop\挑战杯1号语音识别\新\funasr_project
.\.venv\Scripts\python.exe build_external_trainset.py
```

This writes:

```text
data/external_train/
  pos/
  neg/
  pos.jsonl
  neg.jsonl
  phrase_bank.txt
```

Use `data/external_train` for threshold search, small gate training, and phrase
bank construction. Use DataSetA only for stage reporting. Use test set B for
the final organizer run when it becomes available.

## Fair DataSetA Evaluation

Without an external phrase bank:

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root ..\datasetA --no-intent-filter --no-phrase-correct
```

With an external phrase bank:

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root ..\datasetA --phrase-bank data\external_train\phrase_bank.txt
```

The legacy test-label phrase bank is available only for old-result
reproduction:

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root ..\datasetA --use-test-label-phrase-bank
```

Do not use that legacy mode for a fair score.
