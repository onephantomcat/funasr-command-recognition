# P1 → P2 v3_absent_swap

This frozen delivery adds the two datasets that are not present in `v2_b1`:

- B2 ABSENT: 23000 rows. The enrollment speaker is absent from the mixture; `target_present=false`; `target_wav` and `activity_mask` are exact all-zero files.
- B3 enroll-swap: 12000 complete triplets / 36000 rows. Every triplet reuses one bit-identical mixture with target-1, target-2, and absent enrollments.

Counts are deterministic and split-safe. Train/dev/D evaluation material never crosses its source split. Enrollment utterances differ from their target utterances. The target-2 activity mask is computed only from its clean component, never from the mixture.

The files below `assets/` may be hard-linked to the frozen `v2_b1` delivery on this AutoDL data disk to avoid wasting disk space. They are normal readable files and remain valid even if the original pathname changes.

Generator: `p1_v3_absent_swap_builder.v1.0.0`  
Schema: `p1_to_p2.v1`  
Source manifest: `b0ab96c1043eb3221f946f1421ff5bf0ba8c1f2ae99befd5458ce539515990a7`
