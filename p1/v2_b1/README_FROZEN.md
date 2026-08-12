# P1 to P2 v2_b1

- Schema: `p1_to_p2.v1`
- Generator: `p1_v2_b1_builder.v1.0.4`
- Train: 100,000 PRESENT samples
- Dev: 10,000 PRESENT samples
- Frozen confirmation: D_single=2,000, D_overlap=4,000
- Audio: 16 kHz, mono, float32 WAV; primary window 3.6 seconds (57,600 samples)
- Enrollment: 3 seconds
- Activity masks are derived only from clean target audio using the frozen P1 VAD.
- Train/dev/holdout speakers are disjoint. Noise and RIR assets are path-hash partitioned.
- Paths are relative to this directory. Verify SHA256SUMS.txt before use.
- This handoff contains no ABSENT samples; ABSENT belongs to v3_absent_swap.
