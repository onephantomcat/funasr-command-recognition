# P1 reproducibility materials

This directory contains the small, source-controlled part of the P1 data
pipeline. It intentionally excludes raw audio, generated audio, masks, full
manifests, checkpoints, caches, and competition test data.

## Included

- `p1/v2_b1/`: the executable frozen v2 B1 builder, its schema, documented
  parameters, version, and provenance marker.
- `p1/v3_absent_swap/`: the frozen v3 schema and provenance documents.
- `p1/labels/SCENARIOS_AND_ROLES.json`: the public split, scenario, and role
  vocabulary used by the manifests.

## Reproducibility status

- **v2_b1:** builder source is present and passes Python syntax validation.
- **v3_absent_swap:** the original builder source was not preserved in the
  recovered 637 or cloned 651 environments. The checked-in schema and frozen
  provenance are authentic, but this directory must not claim executable or
  bit-exact v3 reproduction until a recovered or reconstructed builder passes
  comparison against the frozen 59,000-row manifest.

## Fair-evaluation boundary

The `train` and `dev` material may be used according to the competition rules.
The `D_single`, `D_overlap`, `D_absent`, and `D_swap` splits are holdout
evaluation material. They must not be used for training, threshold tuning,
model selection, or phrase-bank construction.

## Dependencies

The v2 builder uses Python 3 plus `numpy`, `soundfile`, and `scipy`. Capture the
exact approved environment versions before making a tagged reproducibility
release; this package deliberately does not invent version pins that were not
recorded in the frozen delivery.

Do not add datasets or private machine paths to this directory. Use the parent
repository's license; no new license is asserted here.
