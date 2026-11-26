"""Is the Belt and Road Initiative framed in security or economic terms?

    security framing  cos(BRI, regime-security centroid)
    economic framing  cos(BRI, economic-cooperative centroid)

Any given security-driven reading of the BRI in literature predicts the first exceeds the second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lexicon import BRI_TERMS, ECONOMIC_COOPERATIVE, REGIME_SECURITY, IDEOLOGICAL
from .stats import _rng
from .subordination import semantic_axis, project


@dataclass
class FramingTest:
    term: str
    security_score: float
    economic_score: float
    ideological_score: float
    axis_position: float
    axis_ci: tuple[float, float]

    @property
    def margin(self) -> float:
        """Security minus economic framing. Positive supports the security reading."""
        return self.security_score - self.economic_score

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.margin):
            return "untestable (term absent from vocabulary)"
        return "security-framed" if self.margin > 0 else "economic-framed"


def _centroid_sim(wv, word: str, field: dict[str, str]) -> float:
    vecs = [wv[t] for t in field if t in wv]
    if not vecs or word not in wv:
        return float("nan")
    c = np.mean(vecs, axis=0)
    v = wv[word]
    return float(np.dot(v, c) / (np.linalg.norm(v) * np.linalg.norm(c)))


def framing_test(wv, terms: dict[str, str] = BRI_TERMS,
                 n_boot: int = 1000, seed: int | None = None) -> list[FramingTest]:
    """Score each term on security vs economic framing, with a bootstrap CI.

    `axis_position` places the term on a single security(+) / economic(-) axis;
    the CI resamples the pole vocabularies so the interval reflects the
    arbitrariness of the term lists, not just sampling noise.
    """
    rng = _rng(seed)
    sec = [t for t in REGIME_SECURITY if t in wv]
    eco = [t for t in ECONOMIC_COOPERATIVE if t in wv]
    out = []

    for term in terms:
        if term not in wv:
            out.append(FramingTest(term, float("nan"), float("nan"), float("nan"),
                                   float("nan"), (float("nan"), float("nan"))))
            continue

        base_axis = semantic_axis(wv, sec, eco)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            ax = semantic_axis(
                wv,
                list(rng.choice(sec, size=len(sec), replace=True)),
                list(rng.choice(eco, size=len(eco), replace=True)),
            )
            boots[b] = project(wv, term, ax)

        out.append(FramingTest(
            term=term,
            security_score=_centroid_sim(wv, term, REGIME_SECURITY),
            economic_score=_centroid_sim(wv, term, ECONOMIC_COOPERATIVE),
            ideological_score=_centroid_sim(wv, term, IDEOLOGICAL),
            axis_position=project(wv, term, base_axis),
            axis_ci=(float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))),
        ))
    return out


def neighbour_evidence(wv, term: str, topn: int = 20) -> list[tuple[str, float]]:
    """Nearest neighbours, as qualitative corroboration of the framing test."""
    return wv.most_similar(term, topn=topn) if term in wv else []
