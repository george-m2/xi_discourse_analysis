"""Figures for the paper. APA style, 600 DPI, sized for Word at 1in margins.

Design constraints specific to this deliverable:

* Print-safe. Conference proceedings are often printed in greyscale, so colour
  never carries identity alone: every point is directly labelled and each
  discourse field also has its own marker shape.
* CVD-safe. The categorical palette passes lightness-band, chroma-floor,
  adjacent-pair CVD separation (worst adjacent dE 11.6, protan) and
  normal-vision floor checks. The orange sits below 3:1 contrast on white,
  which is why direct labels are mandatory rather than optional.
* One measure per axis. No dual-axis charts anywhere.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import config  # noqa: E402
from .lexicon import EN  # noqa: E402

# Validated categorical palette. Slot ORDER is load-bearing: the validator checks
# adjacent pairs, and this ordering yields worst-adjacent dE 11.6 (protan) /
# 18.3 (normal vision). Reordering can push a pair below the floor -- re-run
# scripts/validate_palette.js if you change it.
FIELD_STYLE: dict[str, dict[str, str]] = {
    "technocratic":         {"color": "#2166AC", "marker": "o", "label": "Technocratic"},
    "ideological":          {"color": "#B2182B", "marker": "s", "label": "Ideological"},
    "political_governance": {"color": "#762A83", "marker": "P", "label": "Political governance"},
    "regime_security":      {"color": "#E08214", "marker": "^", "label": "Regime security"},
    "ecological":           {"color": "#1B7837", "marker": "D", "label": "Ecological (control)"},
}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

# Neutral slot for any field without a declared style, so an added term set
# degrades to a legible grey series rather than raising mid-figure.
FALLBACK_STYLE = {"color": MUTED, "marker": "v", "label": "Other"}


def _style(field: str) -> dict[str, str]:
    return FIELD_STYLE.get(field, {**FALLBACK_STYLE, "label": field.replace("_", " ").title()})


APA = {
    "font.family": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}


def _apa(ax):
    ax.tick_params(labelsize=9, length=3.5, width=0.8)
    return ax


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _legend(fig, fields: list[str], ncol: int | None = None):
    handles = [
        Line2D([], [], linestyle="none", marker=_style(f)["marker"],
               markerfacecolor=_style(f)["color"], markeredgecolor="black",
               markeredgewidth=0.4, markersize=7, label=_style(f)["label"])
        for f in fields
    ]
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(fields),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))


# --------------------------------------------------------------- vector space

def plot_vector_space(wv, groups: dict[str, list[str]],
                      out: Path | None = None, seed: int = 42) -> Path:
    """PCA and t-SNE projections of the analytical vocabulary.

    Both projections are shown because they answer different questions: PCA is
    linear and deterministic and preserves global structure; t-SNE preserves
    local neighbourhoods but its distances between clusters are not
    interpretable. Neither is evidence on its own -- they illustrate the
    similarity statistics rather than substituting for them.
    """
    out = out or config.FIGURES_DIR / "fig1_vector_space.png"
    words, fields = [], []
    for fname, terms in groups.items():
        for t in terms:
            if t in wv:
                words.append(t)
                fields.append(fname)
    vecs = np.asarray([wv[w] for w in words])

    coords_pca = PCA(n_components=2, random_state=seed).fit_transform(vecs)
    perp = min(max(2, len(words) // 3), len(words) - 1)
    coords_tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                       init="pca", max_iter=2000).fit_transform(vecs)

    explained = PCA(n_components=2, random_state=seed).fit(vecs).explained_variance_ratio_.sum()

    with plt.rc_context(APA):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
        for ax, coords, title in zip(
            axes, [coords_pca, coords_tsne],
            [f"(a) PCA ({explained:.0%} of variance)",
             f"(b) t-SNE (perplexity {perp})"],
        ):
            texts = []
            for i, w in enumerate(words):
                st = _style(fields[i])
                ax.scatter(coords[i, 0], coords[i, 1], c=st["color"],
                           marker=st["marker"], s=48, edgecolors="black",
                           linewidths=0.4, zorder=5)
                texts.append(ax.text(coords[i, 0], coords[i, 1], EN.get(w, w),
                                     fontsize=7, color=INK, zorder=6))
            # These vocabularies cluster tightly, so labels collide badly if
            # placed at a fixed offset. Repel them and draw leader lines back to
            # the marker, which keeps every point identifiable in print.
            adjust_text(
                texts, ax=ax,
                expand=(1.15, 1.3), force_text=(0.4, 0.6), force_static=(0.2, 0.3),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                shrinkA=1, shrinkB=2),
            )
            ax.set_title(title, fontsize=10, pad=6)
            ax.set_xlabel("Dimension 1", fontsize=9)
            ax.set_ylabel("Dimension 2", fontsize=9)
            ax.margins(0.16)
            _apa(ax)
        fig.suptitle("Semantic vector space of Xi Jinping's speeches", fontsize=12, y=1.0)
        _legend(fig, list(groups.keys()))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return _save(fig, out)


# ------------------------------------------------------------------- heatmap

ANNOTATE_MAX = 15   # above this, in-cell numbers collide and become unreadable


def plot_similarity_matrix(wv, terms: list[str], fields: list[str],
                           out: Path | None = None, annotate: bool | None = None) -> Path:
    """Pairwise cosine similarity matrix, blocked by discourse field.

    Sequential single-hue ramp: the quantity is magnitude, not polarity.

    Cell values are printed only for small matrices. At the full term count a
    number in every cell is unreadable, and the figure's job is to show the
    *block structure* -- which fields are internally coherent and which pairs of
    fields overlap. Exact values live in results/tables/t2_cluster_similarities.csv.
    """
    out = out or config.FIGURES_DIR / "fig2_similarity_matrix.png"
    keep = [i for i, t in enumerate(terms) if t in wv]
    terms = [terms[i] for i in keep]
    fields = [fields[i] for i in keep]
    m = np.array([[float(wv.similarity(a, b)) for b in terms] for a in terms])
    labels = [EN.get(t, t) for t in terms]
    n = len(terms)
    if annotate is None:
        annotate = n <= ANNOTATE_MAX

    # field block boundaries, for the separator rules
    bounds = [i for i in range(1, n) if fields[i] != fields[i - 1]]

    size = max(6.0, min(9.5, 0.24 * n + 2.4))
    with plt.rc_context(APA):
        fig, ax = plt.subplots(figsize=(size + 1.1, size))
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), labels, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(n), labels, fontsize=7)

        if annotate:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if m[i, j] > 0.55 else INK)

        # Separator rules make the field blocks legible without any numbers.
        for b in bounds:
            ax.axhline(b - 0.5, color="white", lw=1.6)
            ax.axvline(b - 0.5, color="white", lw=1.6)

        for ticks in (ax.get_yticklabels(), ax.get_xticklabels()):
            for tick, f in zip(ticks, fields):
                tick.set_color(_style(f)["color"])

        cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cb.set_label("Cosine similarity", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title("Pairwise cosine similarity, blocked by discourse field",
                     fontsize=11, pad=10)
        fig.tight_layout()
        return _save(fig, out)


# ---------------------------------------------------------- axis projection

def plot_axis_projection(projections, pos_label: str, neg_label: str,
                         field_of: dict[str, str], out: Path | None = None) -> Path:
    """Dot plot of terms on a bipolar semantic axis, with bootstrap CIs."""
    out = out or config.FIGURES_DIR / "fig3_axis_projection.png"
    projections = sorted(projections, key=lambda p: p.score)
    y = np.arange(len(projections))

    with plt.rc_context(APA):
        fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(projections) + 1.7))
        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for i, p in enumerate(projections):
            st = _style(field_of.get(p.word, "technocratic"))
            ax.plot([p.ci_lo, p.ci_hi], [i, i], color=st["color"], lw=2, zorder=3,
                    solid_capstyle="round")
            ax.scatter(p.score, i, color=st["color"], marker=st["marker"], s=54,
                       edgecolors="white", linewidths=1.2, zorder=4)
        ax.set_yticks(y, [EN.get(p.word, p.word) for p in projections], fontsize=9)
        ax.set_xlabel(f"<-- {neg_label}        Axis position        {pos_label} -->",
                      fontsize=9)
        ax.set_title("Position of administrative vocabulary on the\n"
                     f"{pos_label} / {neg_label} axis", fontsize=11, pad=8)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _apa(ax)
        _legend(fig, sorted({field_of.get(p.word, "technocratic") for p in projections}))
        fig.tight_layout()
        return _save(fig, out)


# ----------------------------------------------------------------- diachronic

def plot_diachronic(series: dict[str, list], ylabel: str,
                    title: str, out: Path | None = None) -> Path:
    """One line per tracked field across year slices. Single measure, one axis.

    Series are direct-labelled at the line end. Where two lines converge the
    labels are nudged apart vertically, since a collision here would make the
    figure ambiguous exactly where the series are hardest to tell apart.
    """
    out = out or config.FIGURES_DIR / "fig4_diachronic.png"
    order = [FIELD_STYLE[f]["color"] for f in FIELD_STYLE]
    markers = [FIELD_STYLE[f]["marker"] for f in FIELD_STYLE]
    n = len(order)

    with plt.rc_context(APA):
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        ends = []
        for i, (name, pts) in enumerate(series.items()):
            xs = [p.year for p in pts]
            ys = [p.value for p in pts]
            ax.plot(xs, ys, color=order[i % n], lw=2, marker=markers[i % n],
                    markersize=5, markeredgecolor="white", markeredgewidth=0.9,
                    label=name, zorder=3)
            if len(xs) > 1:
                ends.append([ys[-1], xs[-1], name, order[i % n]])

        # Separate end labels that would otherwise overprint one another.
        ends.sort()
        span = (max(e[0] for e in ends) - min(e[0] for e in ends)) or 1.0
        min_gap = 0.055 * span
        for k in range(1, len(ends)):
            if ends[k][0] - ends[k - 1][0] < min_gap:
                ends[k][0] = ends[k - 1][0] + min_gap
        for y_lab, x_end, name, colour in ends:
            ax.annotate(name, (x_end, y_lab), fontsize=8, color=colour,
                        textcoords="offset points", xytext=(7, 0), va="center",
                        annotation_clip=False)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.16)
        if len(series) >= 2:
            # Below the axes, so it never lands on the plotted lines or the
            # end labels ("loc=best" collided with both).
            ax.legend(fontsize=8, frameon=False, ncol=len(series),
                      loc="upper center", bbox_to_anchor=(0.5, -0.16))
        _apa(ax)
        fig.tight_layout()
        return _save(fig, out)


# ------------------------------------------------------------------------ BRI

def plot_bri_framing(tests, out: Path | None = None) -> Path:
    """Security vs economic framing scores for Belt and Road terms."""
    out = out or config.FIGURES_DIR / "fig5_bri_framing.png"
    tests = [t for t in tests if np.isfinite(t.security_score)]
    labels = [EN.get(t.term, t.term) for t in tests]
    x = np.arange(len(tests))
    w = 0.24   # thin marks: the comparison is the point, not the ink

    with plt.rc_context(APA):
        fig, ax = plt.subplots(figsize=(5.8, 3.9))
        ax.bar(x - w / 2 - 0.01, [t.security_score for t in tests], w,
               color=_style("regime_security")["color"],
               edgecolor="white", linewidth=1.0, label="Regime-security framing")
        ax.bar(x + w / 2 + 0.01, [t.economic_score for t in tests], w,
               color=_style("technocratic")["color"],
               edgecolor="white", linewidth=1.0, label="Economic-cooperative framing")
        for i, t in enumerate(tests):
            ax.text(i - w / 2 - 0.01, t.security_score + 0.008,
                    f"{t.security_score:.2f}", ha="center", fontsize=8)
            ax.text(i + w / 2 + 0.01, t.economic_score + 0.008,
                    f"{t.economic_score:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylabel("Cosine similarity to field centroid", fontsize=9)
        ax.set_title("Competing framings of the Belt and Road Initiative",
                     fontsize=11, pad=8)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _apa(ax)
        fig.tight_layout()
        return _save(fig, out)


# ------------------------------------------------------------- null vs observed

def plot_against_null(observed: dict[str, tuple[float, float, float]],
                      null_mean: float, null_p95: float,
                      out: Path | None = None) -> Path:
    """Observed mean similarities with bootstrap CIs against the random baseline.

    The baseline band is the point of the figure: a cosine of 0.22 looks
    substantial until it is drawn against a random-pair mean of 0.20.
    """
    out = out or config.FIGURES_DIR / "fig6_against_null.png"
    names = list(observed)
    y = np.arange(len(names))

    with plt.rc_context(APA):
        fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(names) + 1.6))
        ax.axvspan(0, null_p95, color=GRID, alpha=0.55, zorder=1,
                   label="Random-pair baseline (<=95th pct)")
        ax.axvline(null_mean, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2,
                   label=f"Random-pair mean ({null_mean:.3f})")
        for i, n in enumerate(names):
            mean, lo, hi = observed[n]
            ax.plot([lo, hi], [i, i], color="#2166AC", lw=2.4, zorder=3,
                    solid_capstyle="round")
            ax.scatter(mean, i, color="#2166AC", s=56, zorder=4,
                       edgecolors="white", linewidths=1.2)
            ax.text(hi + 0.006, i, f"{mean:.3f}", va="center", fontsize=8)
        ax.set_yticks(y, names, fontsize=9)
        ax.set_xlabel("Mean cosine similarity (95% bootstrap CI)", fontsize=9)
        ax.set_title("Observed similarities against the random-pair baseline",
                     fontsize=11, pad=8)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _apa(ax)
        fig.tight_layout()
        return _save(fig, out)
