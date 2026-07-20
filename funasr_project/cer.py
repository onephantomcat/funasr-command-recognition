# -*- coding: utf-8 -*-
"""Local CER utilities for positive samples only.

The contest organizer's scorer is authoritative. These helpers implement the
standard character-level Levenshtein definition for local debugging:
CER = (substitutions + insertions + deletions) / reference characters.
"""
from command_match import edit_distance
from text_norm import normalize


def _prepare(ref, hyp, do_norm):
    ref = "" if ref is None else str(ref)
    hyp = "" if hyp is None else str(hyp)
    if do_norm:
        ref, hyp = normalize(ref), normalize(hyp)
    return ref, hyp


def cer(ref, hyp, do_norm=False):
    """Return ``(cer, reference_length)`` for one positive sample.

    Negative/rejection samples have no reference transcript and must be scored
    with RR instead. Raising here prevents them from contaminating corpus CER.
    ``do_norm`` is a local debugging option, not a substitute for the official
    contest scorer.
    """
    ref, hyp = _prepare(ref, hyp, do_norm)
    if not ref:
        raise ValueError("CER requires a non-empty positive-sample reference")
    return edit_distance(ref, hyp) / len(ref), len(ref)


def corpus_cer(pairs, do_norm=False):
    """Compute corpus CER from positive ``(reference, hypothesis)`` pairs."""
    total_err, total_len = 0, 0
    for ref, hyp in pairs:
        ref, hyp = _prepare(ref, hyp, do_norm)
        if not ref:
            raise ValueError("Corpus CER received an empty reference")
        total_err += edit_distance(ref, hyp)
        total_len += len(ref)
    if not total_len:
        raise ValueError("Corpus CER requires at least one positive sample")
    return total_err / total_len, total_len
