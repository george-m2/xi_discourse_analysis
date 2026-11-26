"""Word2Vec training"

`seed=42, workers=4`
"""
from __future__ import annotations

import json
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from . import config


class _Progress(CallbackAny2Vec):
    def __init__(self, total: int, desc: str):
        self.pbar = tqdm(total=total, desc=desc, unit="epoch")

    def on_epoch_end(self, model):
        self.pbar.update(1)

    def on_train_end(self, model):
        self.pbar.close()


def model_path(seed: int, tag: str = "primary") -> Path:
    return config.MODEL_DIR / f"w2v_{tag}_seed{seed}"


def train_one(
    sentences: list[list[str]],
    seed: int,
    tag: str = "primary",
    force: bool = False,
    workers: int | None = None,
    **overrides,
) -> Word2Vec:
    path = model_path(seed, tag)
    if path.exists() and not force:
        return Word2Vec.load(str(path))

    params = {**config.W2V_PARAMS, **overrides}
    model = Word2Vec(
        sentences=sentences,
        seed=seed,
        workers=config.WORKERS if workers is None else workers,
        callbacks=[_Progress(params["epochs"], f"{tag} seed={seed}")],
        **params,
    )
    model.save(str(path))
    (path.parent / f"{path.name}.params.json").write_text(
        json.dumps({**params, "seed": seed, "workers": config.WORKERS,
                    "sentences": len(sentences),
                    "tokens": sum(len(s) for s in sentences),
                    "vocab": len(model.wv)}, indent=2),
        encoding="utf-8",
    )
    return model


def train_all(
    sentences: list[list[str]],
    seeds: tuple[int, ...] = config.SEEDS,
    tag: str = "primary",
    force: bool = False,
) -> dict[int, Word2Vec]:
    """Train one model per seed. Returns {seed: model}."""
    return {s: train_one(sentences, s, tag=tag, force=force) for s in seeds}


def load_all(seeds: tuple[int, ...] = config.SEEDS, tag: str = "primary") -> dict[int, Word2Vec]:
    out = {}
    for s in seeds:
        p = model_path(s, tag)
        if p.exists():
            out[s] = Word2Vec.load(str(p))
    if not out:
        raise FileNotFoundError(
            f"No trained models found in {config.MODEL_DIR}. Run: uv run xi-discourse train"
        )
    return out
