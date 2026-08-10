# v3_absent_swap reproducibility status

The frozen delivery identifies generator
`p1_v3_absent_swap_builder.v1.0.0`, source v2 manifest SHA256
`b0ab96c1043eb3221f946f1421ff5bf0ba8c1f2ae99befd5458ce539515990a7`,
and a 59,000-row output manifest SHA256
`03471f980403cc92163b3dbc835b359b939f32b02796586bd5bdaa04b4dac0b7`.

The original generator source was not present in either recovered environment
checked on 2026-08-10. Consequently:

- `README_FROZEN.md`, `SCHEMA.json`, `VERSION`, and `FROZEN` are published as
  authentic provenance and interface documentation.
- No checked-in file currently claims to regenerate v3 bit-for-bit.
- A future recovered or reconstructed builder must reproduce the frozen split
  counts, triplet relations, row fields, output hashes, and final manifest hash
  before this status can be changed to reproducible.

Do not use the frozen `D_absent` or `D_swap` evaluation rows to train, tune, or
select a model while performing that verification.
