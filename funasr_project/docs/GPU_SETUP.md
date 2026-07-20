# GPU Environment Setup

## Verified Local Configuration

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM
- Driver: 610.62, CUDA UMD 13.3
- Python: 3.9 in `.venv`
- PyTorch: `2.7.1+cu118`
- Torchaudio: `2.7.1+cu118`

The driver is backward compatible with the CUDA 11.8 PyTorch runtime. A full
CUDA Toolkit and `nvcc` are not required for FunASR inference.

## Install

Install the general project dependencies first, then replace the CPU PyTorch
build with the CUDA build:

```powershell
cd C:\Users\13238\Desktop\挑战杯1号语音识别\新\funasr_project
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu118.txt
```

## Verify

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output includes `2.7.1+cu118`, `True`, and the NVIDIA GPU name.
`asr_demo.py` and `speaker_verify.py` automatically select CUDA when it is
available, otherwise they safely fall back to CPU.

## Local Smoke Test

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA --limit 20 `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct
```

On the verified local environment, the 20 positive plus 20 negative sample
smoke run completed in 6.7 seconds with 1,031 MB CUDA peak allocation. This is
a diagnostic run, not a final competition score.
