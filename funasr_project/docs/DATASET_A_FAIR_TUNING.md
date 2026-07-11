# DataSetA Fair Tuning Notes

DataSetA is a test set:

- `pos` evaluates target-speaker character error rate (CER).
- `neg` evaluates non-target rejection rate (RR).
- The `识别文本` / `璇嗗埆鏂囨湰` field is label-only.
- Wake audio, wake text, and recognition audio are model inputs.

Therefore, do not train, tune thresholds, build a phrase bank, or correct ASR
output from DataSetA labels.

## External Train/Tuning Set

Build an external set from the local AISHELL subset:

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
bank construction. Use DataSetA only for final reporting.

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
