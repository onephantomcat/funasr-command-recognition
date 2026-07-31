# 3D-Speaker Source Audit — CAM++

Date: 2026-07-30
Role: P4

## Key Findings

1. **Input sample rate**: 16000 Hz (16 kHz) — hardcoded in FBank and load_wav()
2. **Acoustic features**: 80-dim mel-filterbank (FBank) via torchaudio.compliance.kaldi.fbank
   - Mean normalization applied (mean_nor=True in inference)
   - No delta/delta-delta by default
3. **VAD**: NOT applied in the inference pipeline. Full waveform → FBank → model.
   - Training uses WavReader with fixed-duration chunking (default 3.0s)
4. **Long audio**: Processed as single utterance; no built-in chunking or VAD-based cropping
5. **Embedding shape** (for our model `iic/speech_campplus_sv_zh-cn_16k-common`):
   - feat_dim: 80 → FCM output: m_channels * (feat_dim // 8) = 32 * 10 = 320
   - Through TDNN → 3 CAMDenseTDNN blocks (12/24/16 layers) → StatsPool → Dense(embedding_size=192)
   - **embedding_dim = 192** — must verify from actual loaded model
6. **L2 normalization**: Model does NOT L2-normalize the final embedding.
   - Cosine similarity computed at score time via `torch.nn.CosineSimilarity(dim=-1, eps=1e-6)`
7. **Score**: Cosine similarity, range [-1, 1], higher = more similar (same speaker)
8. **Batch inference**: Default script handles only 1-2 files. No batch mode.
9. **Model download**: ModelScope `snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common', revision='v1.0.0')`
10. **License**: Apache License, Version 2.0

## Architecture

```
Waveform (16kHz mono)
  → FBank(80-dim, mean_nor=True) [T, 80]
  → FCM (CNN frontend, feat_dim=80, m_channels=32) [320, T//8]
  → TDNNLayer (128 channels, kernel=5, stride=2)
  → CAMDenseTDNNBlock × 3 (12 layers dil=1, 24 layers dil=2, 16 layers dil=2)
    each with TransitLayer (channel halving)
  → StatsPool (mean + std) [final_channels * 2]
  → DenseLayer (→ 192) 
  → raw embedding (no L2 norm)
```

## Model config for iic/speech_campplus_sv_zh-cn_16k-common

```python
{
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
    'revision': 'v1.0.0',
    'model_pt': 'campplus_cn_common.bin',
}
```

## Verified

- [x] Official GitHub repo (alibaba-damo-academy/3D-Speaker)
- [x] Commit frozen: see COMMIT.txt
- [x] Apache 2.0 License
- [x] Source tree SHA256 saved
- [x] No local modifications
- [x] CAM++ recipe and inference pipeline located
