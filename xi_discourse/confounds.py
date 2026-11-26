"""Controlling for the governance-security link.
Controlling for:
POLYSEMY OF 治理. In this corpus 治理 carries two senses political
   governance (国家治理, 社会治理) and environmental remediation (环境治理,
   水污染治理). In the original model the nearest neighbours of 治理 were
   dominated by the second sense (护岸 revetment, 河势 river regime, 风沙源
   sandstorm source, 岱海 Lake Dai).
POLYSEMY OF 安全. 安全 spans regime security (国家安全, 政治安全, 主权) and
   mundane safety (食品安全, 安全生产, 生态安全). If both 治理 and 安全 lean
   ecological, then that says little about  regime security.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lexicon import ECOLOGICAL


# Sentence-level sense markers. A sentence is ecological if it contains any
# ecological marker and no political marker; political if the reverse.
ECO_MARKERS = frozenset({
    "生态", "环境", "污染", "生态环境", "环境治理", "生态安全", "绿色", "排放",
    "水污染", "大气", "土壤", "河流", "湖泊", "植被", "沙化", "碳排放", "节能",
})
POL_MARKERS = frozenset({
    "党", "中国共产党", "党的领导", "党中央", "政治", "意识形态", "马克思主义",
    "政权", "国家安全", "政治安全", "敌对势力", "国家主权", "执政", "干部",
})


@dataclass
class SensePartition:
    political: list[list[str]]
    ecological: list[list[str]]
    mixed: list[list[str]]
    neither: int

    def summary(self) -> dict[str, int]:
        return {
            "political_sentences": len(self.political),
            "ecological_sentences": len(self.ecological),
            "mixed_sentences": len(self.mixed),
            "unmarked_sentences": self.neither,
        }


def sense_partition(sentences: list[list[str]]) -> SensePartition:
    """Split sentences by discourse register so 治理 can be measured per sense."""
    pol, eco, mix = [], [], []
    neither = 0
    for s in sentences:
        ss = set(s)
        has_e = bool(ss & ECO_MARKERS)
        has_p = bool(ss & POL_MARKERS)
        if has_e and has_p:
            mix.append(s)
        elif has_e:
            eco.append(s)
        elif has_p:
            pol.append(s)
        else:
            neither += 1
    return SensePartition(political=pol, ecological=eco, mixed=mix, neither=neither)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def ecology_direction(wv, terms: dict[str, str] = ECOLOGICAL) -> np.ndarray:
    """Unit vector for the ecological register (centroid of ecological terms)."""
    vecs = [wv[t] for t in terms if t in wv]
    return _unit(np.mean(vecs, axis=0))


def residualise(v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component of `v` lying along `direction`."""
    d = _unit(direction)
    return v - np.dot(v, d) * d


def partial_similarity(wv, a: str, b: str, direction: np.ndarray) -> float:
    """Cosine between a and b after projecting `direction` out of both.
    """
    va, vb = residualise(wv[a], direction), residualise(wv[b], direction)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if not na or not nb:
        return float("nan")
    return float(np.dot(va, vb) / (na * nb))


@dataclass
class MediationResult:
    a: str
    b: str
    raw: float
    residual: float

    @property
    def attenuation(self) -> float:
        """Proportion of the raw similarity removed by the control."""
        return (self.raw - self.residual) / self.raw if self.raw else float("nan")

    @property
    def survives(self) -> bool:
        """Association survives if under half of it is attributable to ecology."""
        return self.residual > 0 and self.attenuation < 0.5


def mediation_check(wv, pairs: list[tuple[str, str]],
                    direction: np.ndarray | None = None) -> list[MediationResult]:
    """Raw vs ecology-residualised similarity for each pair."""
    d = ecology_direction(wv) if direction is None else direction
    out = []
    for a, b in pairs:
        if a not in wv or b not in wv:
            continue
        out.append(MediationResult(
            a=a, b=b,
            raw=float(wv.similarity(a, b)),
            residual=partial_similarity(wv, a, b, d),
        ))
    return out
