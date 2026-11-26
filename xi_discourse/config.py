from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Data locations (src/ is the data directory; see README) ---
DATA_DIR = ROOT / "src"
RAW_CSV = DATA_DIR / "master_xi_speeches.csv"

# --- Derived artefacts ---
CACHE_DIR = ROOT / "src"
CORPUS_CACHE = CACHE_DIR / "corpus_sentences.pkl"
CORPUS_META = CACHE_DIR / "corpus_meta.parquet"
MODEL_DIR = ROOT / "models"
LEGACY_MODEL = DATA_DIR / "word2vec_model"

# --- Outputs ---
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"

# --- Lexicon ---
LEXICON_DIR = ROOT / "lexicon"
USERDICT = LEXICON_DIR / "userdict.txt"
STOPWORDS_FILE = LEXICON_DIR / "stopwords.txt"

# --- Word2Vec hyperparameters ---
# NOTE ON REPRODUCIBILITY: gensim is only deterministic with workers=1.
# With workers>1 the OS thread schedule alters the order of updates, so a fixed
# `seed` does NOT yield identical vectors. All headline numbers are trained at
# WORKERS=1
W2V_PARAMS = dict(
    vector_size=100,
    window=5,
    min_count=10,
    epochs=20,
    sg=1,          # skip-gram
    negative=5,
    sample=1e-4,
)
WORKERS = 1
SEEDS = (42, 43, 44, 45, 46)
PRIMARY_SEED = 42

# --- Statistics ---
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 5000
N_NULL_PAIRS = 20000
RNG_SEED = 20260908

# 2012 (63 docs) and 2026 (65 docs) are partial years and are excluded from
# time-sliced training; every retained year has >=400 documents.
DIACHRONIC_YEARS = tuple(range(2013, 2026))
MIN_DOCS_PER_SLICE = 300

for _d in (MODEL_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LEXICON_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default
