"""Diachronic analysis: tracking discursive shift across Xi's tenure.

The abstract promises "a scalable blueprint for tracking real-time ideological
shifts in political communications". That requires a time dimension, which the
original analysis did not have, although the data supports it: the corpus runs
2012-11-15 to 2026-03-01 with >=480 documents in every full year.

Method. Independently trained embedding spaces are not comparable -- Word2Vec is
identified only up to rotation, so raw coordinates from a 2014 model and a 2023
model mean nothing side by side. We therefore follow the standard alignment
approach (Hamilton, Leskovec & Jurafsky 2016, ACL): train one model per year
slice, then orthogonally align each to a reference year by Procrustes rotation
over the shared vocabulary. Only then are cross-year comparisons meaningful.

Two things are tracked:
    securitisation_series  cos(治理, regime-security centroid) per year
    drift_series           how far each term moves from its own baseline
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from . import config
from .train import train_one


def year_slices(
    sentences: list[list[str]], meta: pd.DataFrame,
    years: tuple[int, ...] = config.DIACHRONIC_YEARS,
    min_docs: int = config.MIN_DOCS_PER_SLICE,
) -> dict[int, list[list[str]]]:
    """Group sentences by publication year, dropping thin slices.

    2012 (63 docs) and 2026 (65 docs) are partial years and are excluded by the
    default year range; the guard here catches any other thin slice.
    """
    out: dict[int, list[list[str]]] = {}
    yr = meta["year"].to_numpy()
    for y in years:
        idx = np.flatnonzero(yr == y)
        if len(idx) == 0:
            continue
        docs = meta.loc[idx, "doc_id"].nunique()
        if docs < min_docs:
            continue
        out[y] = [sentences[i] for i in idx]
    return out


def train_slices(slices: dict[int, list[list[str]]], force: bool = False,
                 min_count: int = 5) -> dict[int, Word2Vec]:
    """One model per year. min_count is lowered because slices are ~1/13 the size."""
    return {
        y: train_one(s, seed=config.PRIMARY_SEED, tag=f"year{y}",
                     force=force, min_count=min_count)
        for y, s in slices.items()
    }


def procrustes_align(base: Word2Vec, other: Word2Vec) -> tuple[np.ndarray, list[str]]:
    """Rotate `other` into `base`'s coordinate frame over shared vocabulary.

    Returns the aligned matrix for the shared vocabulary and the vocabulary list.
    Embedding spaces are identified only up to orthogonal transformation, so this
    rotation is the minimum needed to make two independently trained spaces
    comparable. It preserves all cosine similarities within each space.
    """
    shared = [w for w in base.wv.key_to_index if w in other.wv]
    if len(shared) < 50:
        raise ValueError(f"Only {len(shared)} shared words; alignment unreliable.")
    A = np.asarray([base.wv[w] for w in shared], dtype=np.float64)
    B = np.asarray([other.wv[w] for w in shared], dtype=np.float64)
    A = A - A.mean(0)
    B = B - B.mean(0)
    u, _, vt = np.linalg.svd(B.T @ A)
    R = u @ vt
    return B @ R, shared


def centroid_similarity(wv, word: str, field: list[str]) -> float:
    """Cosine between a word and the centroid of a term field."""
    vecs = [wv[t] for t in field if t in wv]
    if not vecs or word not in wv:
        return float("nan")
    c = np.mean(vecs, axis=0)
    v = wv[word]
    return float(np.dot(v, c) / (np.linalg.norm(v) * np.linalg.norm(c)))


@dataclass
class SeriesPoint:
    year: int
    value: float
    n_sentences: int
    in_vocab: bool


def securitisation_series(
    models: dict[int, Word2Vec], target: str, field: list[str],
    slices: dict[int, list[list[str]]] | None = None,
) -> list[SeriesPoint]:
    """Year-by-year similarity between `target` and a discourse field centroid.

    Because this is a within-model quantity (a cosine computed inside each
    year's own space), it is comparable across years without alignment. Use
    `procrustes_align` only when comparing coordinates or vector displacements.
    """
    out = []
    for y in sorted(models):
        wv = models[y].wv
        out.append(SeriesPoint(
            year=y,
            value=centroid_similarity(wv, target, field),
            n_sentences=len(slices[y]) if slices else 0,
            in_vocab=target in wv,
        ))
    return out


def drift_series(models: dict[int, Word2Vec], target: str,
                 reference_year: int | None = None) -> list[SeriesPoint]:
    """Displacement of `target` from its position in the reference year.

    Reported as 1 - cos(v_year, v_reference) after Procrustes alignment: 0 means
    no movement, larger means the term's usage context has shifted.
    """
    years = sorted(models)
    ref_y = years[0] if reference_year is None else reference_year
    base = models[ref_y]
    if target not in base.wv:
        return [SeriesPoint(y, float("nan"), 0, False) for y in years]

    out = []
    for y in years:
        if target not in models[y].wv:
            out.append(SeriesPoint(y, float("nan"), 0, False))
            continue
        aligned, shared = procrustes_align(base, models[y])
        idx = shared.index(target)
        v_now = aligned[idx]
        v_ref = base.wv[target] - np.mean([base.wv[w] for w in shared], axis=0)
        cos = float(np.dot(v_now, v_ref) /
                    (np.linalg.norm(v_now) * np.linalg.norm(v_ref)))
        out.append(SeriesPoint(y, 1.0 - cos, 0, True))
    return out


def series_frame(series: list[SeriesPoint], name: str) -> pd.DataFrame:
    return pd.DataFrame([{"year": p.year, name: p.value,
                          "in_vocab": p.in_vocab} for p in series])
