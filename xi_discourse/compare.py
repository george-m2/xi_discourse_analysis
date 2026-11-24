"""Comparative machinery: contrast the geometry of two trained corpora.

The abstract offers the method as "a scalable blueprint for ... analysing
authoritarian discourse comparatively at scale". Comparison is what makes an
absolute number interpretable: a cosine of 0.33 between governance and party
vocabulary means little until you know what the same measurement returns on
another corpus.

This module supplies that machinery. It takes any two trained models and
reports, for the same term sets, how field-level geometry differs -- with the
Procrustes alignment that makes two independently trained spaces comparable at
all, and a permutation test on the difference.

WHAT THIS DOES AND DOES NOT DELIVER
-----------------------------------
Implemented and runnable today: comparison between any two corpora, including
the *within-corpus temporal* comparison (early-tenure vs late-tenure Xi), which
the year slices in diachronic.py already provide.

NOT implemented, because it needs data this repository does not contain: a
cross-regime comparison. A claim that Chinese governance discourse is unusually
securitised requires a reference corpus -- pre-Xi People's Daily, another
authoritarian state's official discourse, or a democratic government's -- built
with the same pipeline. `compare_models` accepts any such model; collecting the
corpus is the outstanding work. Until then, comparative claims should be
restricted to the temporal axis, which the data does support.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diachronic import centroid_similarity, procrustes_align
from .stats import _rng


@dataclass
class FieldComparison:
    field_a: str
    field_b: str
    sim_left: float
    sim_right: float
    n_terms: int

    @property
    def delta(self) -> float:
        """Right minus left. Positive means the association is stronger on the right."""
        return self.sim_right - self.sim_left


def compare_models(
    left, right, target: str, fields: dict[str, list[str]], label_left: str = "left",
    label_right: str = "right",
) -> list[FieldComparison]:
    """Compare one term's relation to each discourse field across two models.

    Uses within-model cosines, which are directly comparable across models
    because each is computed inside its own space. No alignment is needed for
    this quantity -- alignment matters only when comparing coordinates or
    displacements (see `displacement`).
    """
    out = []
    for name, terms in fields.items():
        shared = [t for t in terms if t in left.wv and t in right.wv]
        if not shared or target not in left.wv or target not in right.wv:
            continue
        out.append(FieldComparison(
            field_a=label_left, field_b=name,
            sim_left=centroid_similarity(left.wv, target, shared),
            sim_right=centroid_similarity(right.wv, target, shared),
            n_terms=len(shared),
        ))
    return out


def displacement(left, right, target: str) -> float:
    """How far `target` moved between the two spaces, after Procrustes alignment.

    Returned as 1 - cos(aligned_right, left): 0 means no change in usage context.
    """
    if target not in left.wv or target not in right.wv:
        return float("nan")
    aligned, shared = procrustes_align(left, right)
    idx = shared.index(target)
    centre = np.mean([left.wv[w] for w in shared], axis=0)
    v_ref = left.wv[target] - centre
    v_now = aligned[idx]
    return 1.0 - float(np.dot(v_now, v_ref) /
                       (np.linalg.norm(v_now) * np.linalg.norm(v_ref)))


@dataclass
class DeltaTest:
    target: str
    field: str
    delta: float
    p_value: float
    n_permutations: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def delta_significance(
    left, right, target: str, field_terms: list[str],
    n_perm: int = 2000, seed: int | None = None,
) -> DeltaTest:
    """Is the change in `target`'s association with a field larger than chance?

    The null resamples which words constitute the field, so the p-value asks
    whether *this* field's shift is unusual against shifts for arbitrary
    same-sized word sets -- controlling for corpus-wide drift between the two
    models, which would otherwise move every similarity at once.
    """
    shared = [t for t in field_terms if t in left.wv and t in right.wv]
    if not shared or target not in left.wv or target not in right.wv:
        return DeltaTest(target, "", float("nan"), float("nan"), 0)

    observed = (centroid_similarity(right.wv, target, shared)
                - centroid_similarity(left.wv, target, shared))

    pool = [w for w in left.wv.key_to_index if w in right.wv and w != target]
    rng = _rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        sample = [pool[j] for j in rng.integers(0, len(pool), size=len(shared))]
        null[i] = (centroid_similarity(right.wv, target, sample)
                   - centroid_similarity(left.wv, target, sample))

    p = float(np.mean(np.abs(null) >= abs(observed)))
    return DeltaTest(target=target, field="", delta=float(observed),
                     p_value=p, n_permutations=n_perm)


def temporal_contrast(
    year_models: dict[int, object], target: str, fields: dict[str, list[str]],
    early: int | None = None, late: int | None = None,
) -> tuple[list[FieldComparison], dict[str, DeltaTest], float]:
    """Early-tenure vs late-tenure contrast -- the comparison this corpus supports.

    Returns (per-field comparisons, per-field significance tests, displacement).
    """
    years = sorted(year_models)
    e = years[0] if early is None else early
    l = years[-1] if late is None else late
    left, right = year_models[e], year_models[l]

    comps = compare_models(left, right, target, fields,
                           label_left=str(e), label_right=str(l))
    tests = {}
    for name, terms in fields.items():
        t = delta_significance(left, right, target, terms)
        t.field = name
        tests[name] = t
    return comps, tests, displacement(left, right, target)
