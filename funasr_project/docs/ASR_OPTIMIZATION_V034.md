# v0.3.4 ASR Optimization Record

## Guardrails

All optimization choices in this record use public AISHELL-derived data. The
DataSetA labels are not used for threshold selection, phrase-bank construction,
or ASR correction. DataSetA is used only to verify the final candidate once.
Test set B remains the final organizer evaluation.

## Executed Improvement: Output Whitespace Cleanup

Paraformer sometimes emits formatting whitespace between Chinese characters,
for example `开 屏 幕` for `开屏幕`. Those spaces are not lexical content, but a
raw character scorer counts each one as an insertion. `asr_demo.compact_asr_text`
now removes all Unicode whitespace immediately after inference. It does not
look at references, labels, command lists, or test split names.

With the identical full DataSetA configuration (hard CAM++ threshold `0.30`,
no intent filter, no phrase correction, no caches), the local raw-character CER
changed as follows:

| Version | Corpus CER | RR | Positive accept rate | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| v0.3.3, uncleaned output | 119.17% | 91.14% | 69.35% | 618.4 s |
| v0.3.4, whitespace cleanup | 53.43% | 91.14% | 69.35% | 438.9 s |

This is a 55.2% relative reduction in local CER. The elapsed time is a CPU
end-to-end measurement from separate runs and should not be interpreted as an
official efficiency comparison. The v0.3.4 run reported a `3073.22 MB` local
process working set; CUDA was unavailable in this environment.

## Public Robustness Ablation

On the 16-sample public AISHELL-derived overlap development set, the raw
character baseline scored `138.40%` CER in `15.38 s`. The existing
speaker-guided purifier was then tested without DataSetA labels:

| Keep ratio | Floor gain | CER | Elapsed |
| ---: | ---: | ---: | ---: |
| no purification | - | 138.40% | 15.38 s |
| 0.30 | 0.03 | 124.47% | 163.95 s |
| 0.45 | 0.03 | 123.63% | 45.48 s |
| 0.60 | 0.03 | 127.00% | 42.87 s |
| 0.45 | 0.10 | 132.91% | 42.42 s |

`keep_ratio=0.45` and `floor_gain=0.03` are the best tested robust settings.
They reduce public overlap CER by 10.7% relative but roughly triple latency, so
they remain opt-in through `--purify` rather than the default inference path.

## Next Steps

1. Expand the public command-like training corpus and generate clean, overlap,
   babble, and non-target partitions from speaker-disjoint sources.
2. Calibrate the CAM++ acceptance threshold only on that external development
   partition, with a fixed RR constraint; do not search it on DataSetA.
3. Fine-tune or adapt ASR only after a command-domain public corpus is ready;
   accept a candidate only when both clean and overlap external CER improve.
4. Run one frozen-candidate DataSetA evaluation, then package the same
   label-free path through `predict_jsonl.py` for test set B.
