"""Directional measures of subordination.

Cosine similarity is symmetric and measures distributional relatedness, not
hierarchy. Two concepts standing in a subordinating relation ("governance under
party leadership") routinely show only moderate cosine, because they modify one
another rather than substitute for one another. Cosine therefore cannot, on its
own, test the study's actual claim. This module adds three measures that can.

1. SEMANTIC AXIS PROJECTION (after Bolukbasi et al. 2016; Kozlowski, Taddy &
   Evans 2019, *American Sociological Review*). Construct a bipolar axis --
   e.g. regime-security vs market-economic -- and project each administrative
   term onto it. Yields a signed, comparable coordinate rather than a bare
   pairwise number.

2. CONDITIONAL CO-OCCURRENCE ASYMMETRY. Subordination is directional: if
   governance discourse is structurally embedded in security discourse, then
   P(security | governance) should exceed P(governance | security). This is a
   genuine hierarchy test and is invisible to cosine.

3. NEIGHBOURHOOD COMPOSITION. What share of a term's nearest neighbours belong
   to each discourse field, benchmarked against frequency-matched control
   words? Answers "what company does this word keep" directly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .stats import _rng


# ---------------------------------------------------------------- axis projection

def semantic_axis(wv, pole_pos: list[str], pole_neg: list[str]) -> np.ndarray:
    """Unit vector running from the negative pole to the positive pole."""
    pos = np.mean([wv[w] for w in pole_pos if w in wv], axis=0)
    neg = np.mean([wv[w] for w in pole_neg if w in wv], axis=0)
    axis = pos - neg
    return axis / np.linalg.norm(axis)


def project(wv, word: str, axis: np.ndarray) -> float:
    """Cosine of `word` with the axis. Above zero leans to the positive pole."""
    v = wv[word]
    return float(np.dot(v, axis) / np.linalg.norm(v))


@dataclass
class Projection:
    word: str
    score: float
    ci_lo: float
    ci_hi: float


def project_with_ci(
    wv, words: list[str], pole_pos: list[str], pole_neg: list[str],
    n_boot: int = 1000, seed: int | None = None,
) -> list[Projection]:
    """Project `words` onto the axis, bootstrapping over the pole term sets.

    Resampling the pole terms propagates the arbitrariness of dictionary choice
    into the interval, which is the main criticism such term lists attract.
    """
    rng = _rng(seed)
    pos = [w for w in pole_pos if w in wv]
    neg = [w for w in pole_neg if w in wv]
    words = [w for w in words if w in wv]
    boots = np.empty((n_boot, len(words)))
    for b in range(n_boot):
        p = list(rng.choice(pos, size=len(pos), replace=True))
        n = list(rng.choice(neg, size=len(neg), replace=True))
        ax = semantic_axis(wv, p, n)
        boots[b] = [project(wv, w, ax) for w in words]

    base = semantic_axis(wv, pos, neg)
    return [
        Projection(
            word=w,
            score=project(wv, w, base),
            ci_lo=float(np.percentile(boots[:, i], 2.5)),
            ci_hi=float(np.percentile(boots[:, i], 97.5)),
        )
        for i, w in enumerate(words)
    ]


# ------------------------------------------------- conditional co-occurrence

@dataclass
class Asymmetry:
    a: str
    b: str
    p_b_given_a: float
    p_a_given_b: float
    count_a: int
    count_b: int
    count_both: int

    @property
    def asymmetry(self) -> float:
        """P(b|a) - P(a|b). Positive means `a` implies `b` more than the reverse."""
        return self.p_b_given_a - self.p_a_given_b

    @property
    def lift(self) -> float:
        """Ratio of the two conditionals; 1.0 means no directional structure.

        Undefined (NaN) when the terms never co-occur: zero co-occurrence is the
        absence of a directional relation, not an infinitely strong one. Ranking
        by lift without this returns every never-co-occurring pair at the top.
        """
        if self.count_both == 0:
            return float("nan")
        return (self.p_b_given_a / self.p_a_given_b
                if self.p_a_given_b else float("inf"))


def asymmetry_matrix(
    sentences: list[list[str]], set_a: list[str], set_b: list[str]
) -> list[Asymmetry]:
    """All directional pairs between two term sets, computed in a single pass.

    Subordination predicts a positive asymmetry from the subordinate term to the
    dominant one: the subordinate concept cannot be articulated without the
    dominant frame, while the dominant one stands alone.
    """
    terms = set(set_a) | set(set_b)
    single: Counter = Counter()
    pair: Counter = Counter()
    for sent in sentences:
        present = terms.intersection(sent)
        if not present:
            continue
        single.update(present)
        pl = sorted(present)
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                pair[(pl[i], pl[j])] += 1

    out = []
    for a in set_a:
        for b in set_b:
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            both, ca, cb = pair.get(key, 0), single.get(a, 0), single.get(b, 0)
            out.append(Asymmetry(
                a=a, b=b,
                p_b_given_a=both / ca if ca else float("nan"),
                p_a_given_b=both / cb if cb else float("nan"),
                count_a=ca, count_b=cb, count_both=both,
            ))
    return out


def cooccurrence_asymmetry(sentences: list[list[str]], a: str, b: str) -> Asymmetry:
    """Directional co-occurrence for a single pair."""
    return asymmetry_matrix(sentences, [a], [b])[0]


def mean_asymmetry(asyms: list[Asymmetry]) -> float:
    vals = [x.asymmetry for x in asyms if np.isfinite(x.asymmetry)]
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------- neighbourhood composition

@dataclass
class NeighbourhoodProfile:
    word: str
    topn: int
    shares: dict[str, float]
    control_shares: dict[str, float]
    neighbours: list[tuple[str, float]]

    def excess(self, field: str) -> float:
        """Share above the frequency-matched control baseline."""
        return self.shares.get(field, 0.0) - self.control_shares.get(field, 0.0)


def _centroids(wv, fields: dict[str, list[str]]) -> dict[str, np.ndarray]:
    out = {}
    for name, terms in fields.items():
        vecs = [wv[t] for t in terms if t in wv]
        if vecs:
            c = np.mean(vecs, axis=0)
            out[name] = c / np.linalg.norm(c)
    return out


def neighbourhood_profile(
    wv, word: str, fields: dict[str, list[str]], topn: int = 50,
    n_controls: int = 25, freq_tol: float = 0.3, seed: int | None = None,
) -> NeighbourhoodProfile:
    """Assign each of the target's top-n neighbours to its closest discourse field.

    The control baseline uses frequency-matched words: without it a high share
    means nothing, since large fields dominate the neighbourhoods of common
    words regardless of any real relationship.
    """
    cents = _centroids(wv, fields)

    def profile(w: str) -> dict[str, float]:
        counts: Counter = Counter()
        for n, _ in wv.most_similar(w, topn=topn):
            v = wv[n] / np.linalg.norm(wv[n])
            best, best_s = None, 0.0
            for name, c in cents.items():
                s = float(np.dot(v, c))
                if s > best_s:
                    best, best_s = name, s
            if best is not None:
                counts[best] += 1
        return {k: counts.get(k, 0) / topn for k in cents}

    rng = _rng(seed)
    vocab = list(wv.key_to_index.keys())
    logc = np.log10([wv.get_vecattr(w, "count") for w in vocab])
    target = np.log10(wv.get_vecattr(word, "count"))
    cand = [i for i in np.flatnonzero(np.abs(logc - target) <= freq_tol)
            if vocab[i] != word]
    picks = rng.choice(cand, size=min(n_controls, len(cand)), replace=False) if cand else []

    ctrl = [profile(vocab[i]) for i in picks]
    control = {
        k: float(np.mean([p.get(k, 0.0) for p in ctrl])) if ctrl else 0.0
        for k in cents
    }

    return NeighbourhoodProfile(
        word=word, topn=topn, shares=profile(word), control_shares=control,
        neighbours=wv.most_similar(word, topn=15),
    )
