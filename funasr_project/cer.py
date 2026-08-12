# -*- coding: utf-8 -*-
"""Local CER utilities for positive samples only.

The contest organizer's scorer is authoritative. These helpers implement the
standard character-level Levenshtein definition for local debugging:
CER = (substitutions + insertions + deletions) / reference characters.
"""
from dataclasses import dataclass

from text_norm import normalize


@dataclass(frozen=True)
class CerStats:
    """Auditable character-error counts for one sample or a corpus."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_chars: int = 0

    @property
    def errors(self):
        return self.substitutions + self.deletions + self.insertions

    @property
    def value(self):
        if not self.reference_chars:
            raise ValueError("CER requires at least one positive reference character")
        return self.errors / self.reference_chars

    def to_dict(self):
        return {
            "cer": self.value,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "reference_chars": self.reference_chars,
        }

    def __add__(self, other):
        if not isinstance(other, CerStats):
            return NotImplemented
        return CerStats(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_chars=self.reference_chars + other.reference_chars,
        )


def _require_text(value, name):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    return value


def _prepare(ref, hyp, do_norm):
    ref = _require_text(ref, "reference")
    hyp = _require_text(hyp, "hypothesis")
    if do_norm:
        ref, hyp = normalize(ref), normalize(hyp)
    return ref, hyp


def _alignment_counts(ref, hyp):
    """Return deterministic minimum-edit S/D/I counts for two strings."""
    # Each cell stores (total_errors, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, ref_char in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_char in enumerate(hyp, 1):
            if ref_char == hyp_char:
                current.append(previous[j - 1])
                continue

            sub = previous[j - 1]
            delete = previous[j]
            insert = current[j - 1]
            candidates = (
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
            # Prefer substitutions over an equally good delete/insert pairing.
            current.append(min(candidates, key=lambda x: (x[0], x[2] + x[3], x[1], x[2], x[3])))
        previous = current

    _, substitutions, deletions, insertions = previous[-1]
    return CerStats(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_chars=len(ref),
    )


def cer_stats(ref, hyp, do_norm=False):
    """Return auditable S/D/I/N counts for one positive sample."""
    ref, hyp = _prepare(ref, hyp, do_norm)
    if not ref:
        raise ValueError("CER requires a non-empty positive-sample reference")
    return _alignment_counts(ref, hyp)


def corpus_cer_stats(pairs, do_norm=False):
    """Aggregate S/D/I/N over positive pairs without sentence averaging."""
    total = CerStats()
    samples = 0
    for ref, hyp in pairs:
        total += cer_stats(ref, hyp, do_norm=do_norm)
        samples += 1
    if not samples:
        raise ValueError("Corpus CER requires at least one positive sample")
    return total


def cer(ref, hyp, do_norm=False):
    """Return ``(cer, reference_length)`` for one positive sample.

    Negative/rejection samples have no reference transcript and must be scored
    with RR instead. Raising here prevents them from contaminating corpus CER.
    ``do_norm`` is a local debugging option, not a substitute for the official
    contest scorer.
    """
    stats = cer_stats(ref, hyp, do_norm=do_norm)
    return stats.value, stats.reference_chars


def corpus_cer(pairs, do_norm=False):
    """Compute corpus CER from positive ``(reference, hypothesis)`` pairs."""
    stats = corpus_cer_stats(pairs, do_norm=do_norm)
    return stats.value, stats.reference_chars
