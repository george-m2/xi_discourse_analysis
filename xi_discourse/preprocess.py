"""Corpus construction: cleaning, de-duplication, sentence splitting, tokenizing.

Fixes four defects in the original pipeline:

1. BYLINE LEAKAGE. Xinhua bylines -- （新华社塔什干6月23日电 记者骆珺）-- appear in
   7,986 of 14,545 documents. Stripping only non-Chinese characters glued
   journalist names into the vocabulary (e.g. 杜尚泽蒋 sat at cosine 0.58 from
   安全 in the original model).

2. IN-DOCUMENT DUPLICATION. The scraper collects both a summary <p> and the body
   paragraphs, so ~45% of sentence content is repeated within a document,
   affecting 94% of documents. This double-counts co-occurrences for the
   duplicated span and inflates the token total.

3. NO SENTENCE BOUNDARIES. The original stripped all punctuation before
   segmenting, so jieba segmented across sentence boundaries (producing junk
   tokens such as 就实, 这步, 以守, 心政) and Word2Vec's window slid across whole
   871-token documents rather than stopping at sentence edges.

4. SINGLE-CHARACTER FILTER. `len(t) > 1` silently removed 党 (party) -- the
   central ideological term -- from the model entirely.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import jieba
import pandas as pd
from tqdm import tqdm

from . import config
from .lexicon import SINGLE_CHAR_KEEP

# （新华社北京6月23日电 记者骆珺）/ （记者张晓松）/ 新华社记者... trailing agency credits
BYLINE_RE = re.compile(
    r"[（(]\s*(?:新华社|中新社|人民日报|本报)[^）)]{0,60}[）)]"
    r"|[（(]\s*记者[^）)]{0,60}[）)]"
    r"|新华社记者[\u4e00-\u9fff·、 ]{2,30}"
    r"|[（(]\s*完\s*[）)]"
)
SENTENCE_SPLIT_RE = re.compile(r"[。！？；!?;\n\r]+")
NON_CHINESE_RE = re.compile(r"[^\u4e00-\u9fff]")

_JIEBA_READY = False


def init_jieba() -> None:
    """Load the user dictionary. Idempotent."""
    global _JIEBA_READY
    if _JIEBA_READY:
        return
    if config.USERDICT.exists():
        jieba.load_userdict(str(config.USERDICT))
    _JIEBA_READY = True


def load_stopwords() -> set[str]:
    if not config.STOPWORDS_FILE.exists():
        return set()
    lines = config.STOPWORDS_FILE.read_text(encoding="utf-8").splitlines()
    return {w.strip() for w in lines if w.strip() and not w.startswith("#")}


def strip_bylines(text: str) -> str:
    return BYLINE_RE.sub("", text)


def dedupe_sentences(sentences: list[str]) -> list[str]:
    """Drop repeated sentences within a document, preserving first occurrence.

    Short sentences (<8 chars) are exempt: formulaic openers legitimately recur
    and removing them would distort local context.
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in sentences:
        if len(s) >= 8:
            if s in seen:
                continue
            seen.add(s)
        out.append(s)
    return out


def split_sentences(text: str) -> list[str]:
    """Split into sentences, then strip non-Chinese characters within each."""
    raw = SENTENCE_SPLIT_RE.split(text)
    out = []
    for s in raw:
        s = NON_CHINESE_RE.sub("", s)
        if len(s) >= 4:          # drop fragments too short to give context
            out.append(s)
    return out


def tokenize(sentence: str, stopwords: set[str]) -> list[str]:
    """Segment one sentence; keep multi-char tokens plus whitelisted single chars."""
    toks = jieba.lcut(sentence)
    return [
        t for t in toks
        if t not in stopwords and (len(t) > 1 or t in SINGLE_CHAR_KEEP)
    ]


@dataclass
class CorpusStats:
    documents_raw: int
    documents_used: int
    sentences: int
    tokens: int
    tokens_before_dedupe: int
    mean_sentence_len: float
    date_min: str
    date_max: str


def build_corpus(
    csv_path: Path | None = None,
    cache: Path | None = None,
    force: bool = False,
) -> tuple[list[list[str]], pd.DataFrame, CorpusStats]:
    """Build the sentence-level corpus.

    Returns
    -------
    sentences : list[list[str]]
        Tokenized sentences -- the training unit for Word2Vec.
    meta : DataFrame
        One row per sentence with `doc_id` and `year`, for time slicing.
    stats : CorpusStats
    """
    csv_path = csv_path or config.RAW_CSV
    cache = cache or config.CORPUS_CACHE

    if cache.exists() and not force:
        with open(cache, "rb") as f:
            payload = pickle.load(f)
        return payload["sentences"], payload["meta"], CorpusStats(**payload["stats"])

    init_jieba()
    stopwords = load_stopwords()

    df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
    documents_raw = len(df)
    df = df.dropna(subset=["text"]).copy()
    df["date"] = pd.to_datetime(df["publication"], errors="coerce")

    # Drop documents that duplicate another document wholesale (39 in the corpus).
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    sentences: list[list[str]] = []
    rows: list[dict] = []
    tokens_before = 0

    for doc_id, (text, date) in enumerate(
        tqdm(list(zip(df["text"], df["date"])), desc="Preprocessing", unit="doc")
    ):
        raw_sents = split_sentences(strip_bylines(str(text)))
        tokens_before += sum(len(s) for s in raw_sents)
        for sent in dedupe_sentences(raw_sents):
            toks = tokenize(sent, stopwords)
            if len(toks) < 2:      # a 1-token sentence carries no co-occurrence
                continue
            sentences.append(toks)
            rows.append({"doc_id": doc_id, "year": date.year if pd.notna(date) else None})

    meta = pd.DataFrame(rows)
    n_tokens = sum(len(s) for s in sentences)
    stats = CorpusStats(
        documents_raw=documents_raw,
        documents_used=len(df),
        sentences=len(sentences),
        tokens=n_tokens,
        tokens_before_dedupe=tokens_before,
        mean_sentence_len=n_tokens / max(len(sentences), 1),
        date_min=str(df["date"].min().date()),
        date_max=str(df["date"].max().date()),
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"sentences": sentences, "meta": meta, "stats": asdict(stats)}, f)

    return sentences, meta, stats
