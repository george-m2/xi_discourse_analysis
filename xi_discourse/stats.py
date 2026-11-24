"""Inferential statistics for cosine similarities.

A bare cosine of 0.38 is uninterpretable: cosine similarities in a Word2Vec
space have a non-zero, frequency-dependent baseline. Random word pairs in this
corpus average ~0.20. Everything here is therefore reported against an explicit
null, with uncertainty attached.

Provides:
    null_distribution      random-pair and frequency-matched baselines
    z_against_null         standardised effect size for an observed mean
    bootstrap_ci           CI on a mean similarity over term pairs
    permutation_cluster_test   is a labelled split more separable than chance?
    seed_spread            mean +/- sd of a similarity across independent seeds
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(config.RNG_SEED if seed is None else seed)


@dataclass
class Null:
    """Baseline distribution of cosine similarity between unrelated words."""
    mean: float
    sd: float
    p95: float
    p99: float
    n: int
    kind: str

    def z(self, value: float) -> float:
        return (value - self.mean) / self.sd if self.sd else float("nan")


def null_distribution(
    wv,
    kind: str = "vocab",
    n_pairs: int = config.N_NULL_PAIRS,
    freq_pool: int = 2000,
    seed: int | None = None,
) -> Null:
    """Baseline cosine distribution.

    kind='vocab'      : pairs drawn from the whole vocabulary
    kind='frequency'  : pairs drawn from the `freq_pool` most frequent words,
                        which matches the frequency profile of the analytical
                        terms and controls for the fact that rare words sit in
                        a sparser, lower-similarity region of the space.
    """
    vocab = list(wv.key_to_index.keys())
    pool = vocab if kind == "vocab" else vocab[:freq_pool]
    rng = _rng(seed)
    idx = rng.integers(0, len(pool), size=(n_pairs, 2))
    sims = np.array([
        float(wv.similarity(pool[i], pool[j])) for i, j in idx if i != j
    ])
    return Null(
        mean=float(sims.mean()), sd=float(sims.std()),
        p95=float(np.percentile(sims, 95)), p99=float(np.percentile(sims, 99)),
        n=len(sims), kind=kind,
    )


def frequency_stratified_null(
    wv, terms: list[str], n_pairs: int = 5000, tol: float = 0.35, seed: int | None = None
) -> Null:
    """Null built from words frequency-matched to `terms` themselves.

    For each analytical term we sample control words whose log-frequency is
    within `tol` dex, then measure similarities among those controls. This is
    the strictest baseline: it asks whether the observed similarities exceed
    what words of *this particular frequency profile* show by default.
    """
    counts = np.array([wv.get_vecattr(w, "count") for w in wv.key_to_index])
    vocab = list(wv.key_to_index.keys())
    logc = np.log10(counts)
    rng = _rng(seed)

    matched: list[str] = []
    for t in terms:
        target = np.log10(wv.get_vecattr(t, "count"))
        cand = np.flatnonzero(np.abs(logc - target) <= tol)
        if len(cand) == 0:
            continue
        take = rng.choice(cand, size=min(200, len(cand)), replace=False)
        matched.extend(vocab[i] for i in take if vocab[i] not in terms)

    if len(matched) < 2:
        return Null(np.nan, np.nan, np.nan, np.nan, 0, "frequency_stratified")

    idx = rng.integers(0, len(matched), size=(n_pairs, 2))
    sims = np.array([
        float(wv.similarity(matched[i], matched[j])) for i, j in idx if i != j
    ])
    return Null(
        mean=float(sims.mean()), sd=float(sims.std()),
        p95=float(np.percentile(sims, 95)), p99=float(np.percentile(sims, 99)),
        n=len(sims), kind="frequency_stratified",
    )


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = config.N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean. Returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = _rng(seed)
    boots = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


@dataclass
class PermutationResult:
    observed: float
    null_mean: float
    null_sd: float
    p_value: float
    n_permutations: int

    @property
    def separable(self) -> bool:
        """True if the labelled split separates better than chance (p < .05)."""
        return self.p_value < 0.05


def permutation_cluster_test(
    wv, group_a: list[str], group_b: list[str],
    n_perm: int = config.N_PERMUTATIONS, seed: int | None = None,
) -> PermutationResult:
    """Test whether the a/b labelling separates the terms better than chance.

    Statistic = mean(intra-group similarity) - mean(cross-group similarity).
    A HIGH value with a LOW p-value means the two sets occupy distinguishable
    regions -- i.e. the vocabularies are semantically SEPARATE. A value near the
    null with a high p-value means the labelling carries no structure -- i.e.
    the vocabularies are entangled.

    Note the direction carefully: for this study, *entanglement* is the finding
    that supports subordination, and *separability* is the finding that
    supports the technocratic-autonomy reading.
    """
    words = list(group_a) + list(group_b)
    na = len(group_a)
    sim = {}

    def c(x: str, y: str) -> float:
        key = (x, y) if x < y else (y, x)
        if key not in sim:
            sim[key] = float(wv.similarity(x, y))
        return sim[key]

    def stat(a: list[str], b: list[str]) -> float:
        intra = ([c(a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a))] +
                 [c(b[i], b[j]) for i in range(len(b)) for j in range(i + 1, len(b))])
        cross = [c(x, y) for x in a for y in b]
        if not intra or not cross:
            return float("nan")
        return float(np.mean(intra) - np.mean(cross))

    observed = stat(list(group_a), list(group_b))
    rng = _rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(len(words))
        null[k] = stat([words[i] for i in p[:na]], [words[i] for i in p[na:]])

    return PermutationResult(
        observed=observed, null_mean=float(np.nanmean(null)),
        null_sd=float(np.nanstd(null)),
        p_value=float(np.nanmean(null >= observed)), n_permutations=n_perm,
    )


def seed_spread(models: dict[int, object], a: str, b: str) -> tuple[float, float, int]:
    """(mean, sd, n) of cos(a,b) across independently seeded models."""
    vals = [float(m.wv.similarity(a, b)) for m in models.values()
            if a in m.wv and b in m.wv]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def cross_set_similarities(wv, set_a: list[str], set_b: list[str]) -> list[float]:
    return [float(wv.similarity(a, b)) for a in set_a for b in set_b if a != b]


def intra_set_similarities(wv, terms: list[str]) -> list[float]:
    return [float(wv.similarity(terms[i], terms[j]))
            for i in range(len(terms)) for j in range(i + 1, len(terms))]
