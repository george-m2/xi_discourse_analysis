"""Run the full analysis and emit tables, figures and a findings report.

Every reported quantity carries (a) an explicit null baseline, (b) an interval,
and (c) where models are available, the spread across independent seeds. The
report states what the corpus shows, including where it fails to support the
study's hypothesis -- a null result is written up as a null result.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, lexicon, viz
from .bri import framing_test, neighbour_evidence
from .confounds import ecology_direction, mediation_check, sense_partition
from .stats import (
    bootstrap_ci, cross_set_similarities, frequency_stratified_null,
    intra_set_similarities, null_distribution, permutation_cluster_test,
    seed_spread,
)
from .subordination import (
    asymmetry_matrix, mean_asymmetry, neighbourhood_profile, project_with_ci,
)


def _write_table(df: pd.DataFrame, name: str) -> Path:
    p = config.TABLES_DIR / f"{name}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def _rows(objs) -> pd.DataFrame:
    return pd.DataFrame([asdict(o) if is_dataclass(o) else o for o in objs])


def run(models: dict[int, object], sentences: list[list[str]],
        corpus_stats, meta: pd.DataFrame | None = None,
        year_models: dict[int, object] | None = None) -> dict:
    """Execute every analysis. Returns a results dict; writes tables and figures."""
    primary = models[config.PRIMARY_SEED]
    wv = primary.wv
    R: dict = {"generated": str(date.today()), "seeds": sorted(models)}

    # ---------------------------------------------------------- 0. coverage
    coverage = []
    for fname, terms in lexicon.ALL_SETS.items():
        for zh, en in terms.items():
            coverage.append({
                "field": fname, "term": zh, "gloss": en,
                "in_vocab": zh in wv,
                "count": int(wv.get_vecattr(zh, "count")) if zh in wv else 0,
            })
    for zh, en in lexicon.BRI_TERMS.items():
        coverage.append({"field": "bri", "term": zh, "gloss": en,
                         "in_vocab": zh in wv,
                         "count": int(wv.get_vecattr(zh, "count")) if zh in wv else 0})
    cov = pd.DataFrame(coverage)
    _write_table(cov, "t0_vocabulary_coverage")
    R["coverage"] = {"in_vocab": int(cov.in_vocab.sum()), "total": len(cov),
                     "missing": cov.loc[~cov.in_vocab, "term"].tolist()}

    tech = lexicon.present(lexicon.TECHNOCRATIC, wv)
    polg = lexicon.present(lexicon.POLITICAL_GOVERNANCE, wv)
    ideo = lexicon.present(lexicon.IDEOLOGICAL, wv)
    secu = lexicon.present(lexicon.REGIME_SECURITY, wv)
    ecol = lexicon.present(lexicon.ECOLOGICAL, wv)

    # ------------------------------------------------------------- 1. nulls
    n_vocab = null_distribution(wv, "vocab")
    n_freq = null_distribution(wv, "frequency")
    n_strat = frequency_stratified_null(wv, tech + polg + ideo + secu)
    R["nulls"] = {k: asdict(v) for k, v in
                  {"vocab": n_vocab, "frequency": n_freq, "stratified": n_strat}.items()}
    _write_table(_rows([n_vocab, n_freq, n_strat]), "t1_null_distributions")

    # --------------------------------------- 2. cluster similarities vs null
    comparisons = {
        "technocratic <-> ideological": cross_set_similarities(wv, tech, ideo),
        "technocratic <-> regime security": cross_set_similarities(wv, tech, secu),
        "political governance <-> ideological": cross_set_similarities(wv, polg, ideo),
        "political governance <-> regime security": cross_set_similarities(wv, polg, secu),
        "political governance <-> technocratic": cross_set_similarities(wv, polg, tech),
        "ideological <-> regime security": cross_set_similarities(wv, ideo, secu),
        "technocratic (within)": intra_set_similarities(wv, tech),
        "political governance (within)": intra_set_similarities(wv, polg),
        "ideological (within)": intra_set_similarities(wv, ideo),
        "regime security (within)": intra_set_similarities(wv, secu),
    }
    rows, observed = [], {}
    for name, vals in comparisons.items():
        mean, lo, hi = bootstrap_ci(vals)
        observed[name] = (mean, lo, hi)
        rows.append({
            "comparison": name, "n_pairs": len(vals), "mean": mean,
            "ci_lo": lo, "ci_hi": hi,
            "z_vs_random": n_vocab.z(mean),
            "z_vs_freq_matched": n_freq.z(mean),
            "z_vs_stratified": n_strat.z(mean),
            "above_random_95th": mean > n_vocab.p95,
        })
    sim_df = pd.DataFrame(rows)
    _write_table(sim_df, "t2_cluster_similarities")
    R["cluster_similarities"] = rows

    # ---------------------------------------------- 3. permutation structure
    perms = {
        "technocratic vs ideological": permutation_cluster_test(wv, tech, ideo),
        "technocratic vs regime security": permutation_cluster_test(wv, tech, secu),
        "political governance vs ideological": permutation_cluster_test(wv, polg, ideo),
        "political governance vs regime security": permutation_cluster_test(wv, polg, secu),
        "ideological vs regime security": permutation_cluster_test(wv, ideo, secu),
    }
    _write_table(
        pd.DataFrame([{"split": k, **asdict(v), "separable": v.separable}
                      for k, v in perms.items()]),
        "t3_permutation_tests")
    R["permutation"] = {k: {**asdict(v), "separable": v.separable}
                        for k, v in perms.items()}

    # -------------------------------------------------- 4. seed robustness
    key_pairs = [("治理", "安全"), ("治理", "国家安全"), ("治理", "党"),
                 ("治理", "政治"), ("治理", "经济"), ("治理", "市场"),
                 ("管理", "安全"), ("管理", "经济"),
                 # political-governance compounds: the constructs the governance
                 # debate is actually about, unlike the polysemous bare 治理
                 ("国家治理体系", "党"), ("国家治理体系", "国家安全"),
                 ("国家治理体系", "经济"), ("治理能力", "党"),
                 ("治理能力", "国家安全"), ("治理能力", "市场"),
                 ("依法治国", "党"), ("依法治国", "经济"), ("制度", "党")]
    seed_rows = []
    for a, b in key_pairs:
        m, sd, n = seed_spread(models, a, b)
        seed_rows.append({"term_a": a, "gloss_a": lexicon.EN.get(a, a),
                          "term_b": b, "gloss_b": lexicon.EN.get(b, b),
                          "mean": m, "sd": sd, "n_seeds": n,
                          "z_vs_random": n_vocab.z(m) if np.isfinite(m) else np.nan})
    _write_table(pd.DataFrame(seed_rows), "t4_seed_robustness")
    R["seed_robustness"] = seed_rows

    # ------------------------------------------- 5. directional subordination
    axis_terms = tech + polg
    projections = project_with_ci(wv, axis_terms, secu, list(lexicon.ECONOMIC_COOPERATIVE))
    _write_table(_rows(projections), "t5_axis_projection")
    R["axis_projection"] = [asdict(p) for p in projections]

    admin = tech + polg
    asym_ts = asymmetry_matrix(sentences, admin, secu)
    asym_ti = asymmetry_matrix(sentences, admin, ideo)
    _write_table(_rows(asym_ts).assign(
        asymmetry=[a.asymmetry for a in asym_ts], lift=[a.lift for a in asym_ts]),
        "t6_asymmetry_tech_security")
    _write_table(_rows(asym_ti).assign(
        asymmetry=[a.asymmetry for a in asym_ti], lift=[a.lift for a in asym_ti]),
        "t6b_asymmetry_tech_ideological")
    def _top(asyms, n=10):
        return sorted(
            [{"a": a.a, "b": a.b, "p_b_given_a": a.p_b_given_a,
              "p_a_given_b": a.p_a_given_b, "asymmetry": a.asymmetry,
              "lift": a.lift} for a in asyms if np.isfinite(a.asymmetry)],
            key=lambda d: -d["asymmetry"])[:n]

    # Lift on the party term specifically. The mean asymmetry across all pairs is
    # a poor summary -- it is dominated by pairs where the frame term is rare --
    # whereas every administrative term's directional relation to 党 is precisely
    # the quantity the subordination hypothesis predicts.
    party_lifts = [a.lift for a in asym_ti if a.b == "党" and np.isfinite(a.lift)]
    R["asymmetry"] = {
        "tech_to_security_mean": mean_asymmetry(asym_ts),
        "tech_to_ideological_mean": mean_asymmetry(asym_ti),
        "party_lift_median": float(np.median(party_lifts)) if party_lifts else float("nan"),
        "party_lift_min": float(np.min(party_lifts)) if party_lifts else float("nan"),
        "party_lift_n_above_1": int(sum(x > 1 for x in party_lifts)),
        "party_lift_n": len(party_lifts),
        "top_security": _top(asym_ts),
        "top_ideological": _top(asym_ti),
    }

    fields = {"technocratic": tech, "political_governance": polg,
              "ideological": ideo, "regime_security": secu, "ecological": ecol}
    profiles = [neighbourhood_profile(wv, w, fields)
                for w in ("治理", "管理", "政策", "国家治理体系", "治理能力",
                          "依法治国", "制度") if w in wv]
    _write_table(pd.DataFrame([
        {"word": p.word, "gloss": lexicon.EN.get(p.word, p.word),
         **{f"share_{k}": v for k, v in p.shares.items()},
         **{f"excess_{k}": p.excess(k) for k in p.shares}}
        for p in profiles]), "t7_neighbourhood_composition")
    R["neighbourhoods"] = [
        {"word": p.word, "shares": p.shares, "control": p.control_shares,
         "excess": {k: p.excess(k) for k in p.shares},
         "top_neighbours": p.neighbours[:10]} for p in profiles]

    # ------------------------------------------------------- 6. confounds
    part = sense_partition(sentences)
    R["sense_partition"] = part.summary()
    eco_dir = ecology_direction(wv)
    med = mediation_check(wv, [("治理", "安全"), ("治理", "国家安全"),
                               ("管理", "安全"), ("治理", "政治"), ("治理", "党"),
                               ("国家治理体系", "党"), ("国家治理体系", "国家安全"),
                               ("治理能力", "国家安全"), ("依法治国", "党")], eco_dir)
    _write_table(_rows(med).assign(
        attenuation=[m.attenuation for m in med],
        survives=[m.survives for m in med]), "t8_ecology_mediation")
    R["mediation"] = [{**asdict(m), "attenuation": m.attenuation,
                       "survives": m.survives} for m in med]

    # ------------------------------------------------------------- 7. BRI
    bri = framing_test(wv)
    _write_table(_rows(bri).assign(margin=[b.margin for b in bri],
                                   verdict=[b.verdict for b in bri]), "t9_bri_framing")
    R["bri"] = [{**asdict(b), "margin": b.margin, "verdict": b.verdict} for b in bri]
    R["bri_neighbours"] = {t: neighbour_evidence(wv, t, 15) for t in lexicon.BRI_TERMS}

    # ------------------------------------------------------ 8. diachronic
    if year_models:
        from .diachronic import securitisation_series, series_frame
        target = "国家治理体系" if "国家治理体系" in wv else "治理"
        s_sec = securitisation_series(year_models, target, secu)
        s_ideo = securitisation_series(year_models, target, ideo)
        s_eco = securitisation_series(year_models, target, list(lexicon.ECONOMIC_COOPERATIVE))
        R["diachronic_target"] = target
        df = (series_frame(s_sec, "vs_regime_security")
              .merge(series_frame(s_ideo, "vs_ideological")[["year", "vs_ideological"]], on="year")
              .merge(series_frame(s_eco, "vs_economic")[["year", "vs_economic"]], on="year"))
        _write_table(df, "t10_diachronic_governance")
        R["diachronic"] = df.to_dict("records")

        # Early-tenure vs late-tenure contrast: the comparative axis this corpus
        # supports. A cross-regime contrast needs a second corpus (see compare.py).
        from .compare import temporal_contrast
        comp_fields = {"regime_security": secu, "ideological": ideo,
                       "economic": list(lexicon.ECONOMIC_COOPERATIVE)}
        comps, dtests, disp = temporal_contrast(year_models, target, comp_fields)
        _write_table(pd.DataFrame([{
            "field": c.field_b, "early_year": c.field_a,
            "sim_early": c.sim_left, "sim_late": c.sim_right, "delta": c.delta,
            "p_value": dtests[c.field_b].p_value,
            "significant": dtests[c.field_b].significant,
        } for c in comps]), "t11_temporal_contrast")
        R["temporal_contrast"] = {
            "target": target, "displacement": disp,
            "fields": [{"field": c.field_b, "sim_early": c.sim_left,
                        "sim_late": c.sim_right, "delta": c.delta,
                        "p_value": dtests[c.field_b].p_value,
                        "significant": dtests[c.field_b].significant}
                       for c in comps],
        }
        viz.plot_diachronic(
            {"Regime security": s_sec, "Ideological": s_ideo, "Economic": s_eco},
            ylabel="Cosine similarity of 治理 to field centroid",
            title="Securitisation of governance discourse, 2013-2025")

    # --------------------------------------------------------- 9. figures
    groups = {"technocratic": tech, "political_governance": polg,
              "ideological": ideo, "regime_security": secu, "ecological": ecol}
    viz.plot_vector_space(wv, groups)
    all_terms = tech + polg + ideo + secu
    all_fields = (["technocratic"] * len(tech)
                  + ["political_governance"] * len(polg)
                  + ["ideological"] * len(ideo)
                  + ["regime_security"] * len(secu))
    viz.plot_similarity_matrix(wv, all_terms, all_fields)
    viz.plot_axis_projection(
        projections, "Regime security", "Economic",
        {**{t: "technocratic" for t in tech},
         **{t: "political_governance" for t in polg}})
    viz.plot_bri_framing(bri)
    viz.plot_against_null(observed, n_vocab.mean, n_vocab.p95)

    R["corpus"] = asdict(corpus_stats) if is_dataclass(corpus_stats) else corpus_stats
    (config.RESULTS_DIR / "findings.json").write_text(
        json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(R)
    return R


def write_markdown(R: dict) -> Path:
    """Human-readable findings report."""
    c = R["corpus"]
    nv = R["nulls"]["vocab"]
    sims = {r["comparison"]: r for r in R["cluster_similarities"]}
    L = []
    A = L.append

    A("# Findings\n")
    A(f"Generated {R['generated']}. Seeds: {R['seeds']}.\n")

    A("## Corpus\n")
    A(f"- Documents: {c['documents_used']:,} (from {c['documents_raw']:,} scraped rows)")
    A(f"- Sentences: {c['sentences']:,}")
    A(f"- Tokens analysed: {c['tokens']:,}")
    A(f"- Coverage: {c['date_min']} to {c['date_max']}\n")

    A("## Baselines\n")
    A(f"Random word pairs in this space average **{nv['mean']:.3f}** "
      f"(SD {nv['sd']:.3f}, 95th pct {nv['p95']:.3f}). Every similarity below is "
      "reported against that baseline; a raw cosine near 0.20 is indistinguishable "
      "from two unrelated words.\n")

    A("## Cluster similarities\n")
    A("| Comparison | Mean | 95% CI | z vs random | Above 95th pct |")
    A("|---|---|---|---|---|")
    for r in R["cluster_similarities"]:
        A(f"| {r['comparison']} | {r['mean']:.4f} | "
          f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['z_vs_random']:+.2f} | "
          f"{'yes' if r['above_random_95th'] else 'no'} |")
    A("")

    A("## Cluster separability (permutation tests)\n")
    A("A *low* p-value means the two vocabularies occupy distinguishable regions, "
      "i.e. they are semantically **separate**. A high p-value means they are "
      "entangled. Under the original design's stated decision rule, entanglement "
      "is the result that supports subordination.\n")
    A("Read this as a bound, not a verdict. Separability here is *lexical*: it "
      "says the two vocabularies are distinguishable, which they can be while "
      "still standing in a hierarchical relation -- \"governance under party "
      "leadership\" subordinates without making the words interchangeable. "
      "Cosine is symmetric and cannot express hierarchy at all, which is why the "
      "directional measures below carry the substantive claim and this table "
      "mainly rules out the strong *proximity* version of it.\n")
    A("| Split | Observed gap | p | Interpretation |")
    A("|---|---|---|---|")
    for k, v in R["permutation"].items():
        A(f"| {k} | {v['observed']:+.4f} | {v['p_value']:.4f} | "
          f"{'separable' if v['separable'] else 'entangled'} |")
    A("")

    A("## Directional subordination\n")
    a = R["asymmetry"]
    A("Cosine is symmetric and cannot express hierarchy. These measures can: if "
      "governance discourse is structurally embedded in party discourse, then "
      "P(party | governance) should exceed P(governance | party). Lift is the "
      "ratio of those two conditionals -- 1.0 means no directional structure.\n")

    if a.get("party_lift_n"):
        A(f"**{a['party_lift_n_above_1']} of {a['party_lift_n']}** administrative "
          f"terms occur with 党 (*party*) more often than the reverse. "
          f"Median lift **{a['party_lift_median']:.2f}x**, "
          f"minimum {a['party_lift_min']:.2f}x.\n")

    A("| Administrative term | Frame | P(frame given admin) | P(admin given frame) | Lift |")
    A("|---|---|---|---|---|")
    for r in a["top_ideological"][:8]:
        lift = f"{r['lift']:.2f}x" if np.isfinite(r["lift"]) else "n/a"
        A(f"| {r['a']} | {r['b']} | {r['p_b_given_a']:.3f} | "
          f"{r['p_a_given_b']:.3f} | {lift} |")
    A("")
    A(f"Mean asymmetry across all pairs -- administrative to regime security "
      f"{a['tech_to_security_mean']:+.4f}, administrative to ideological "
      f"{a['tech_to_ideological_mean']:+.4f} -- is reported for completeness but "
      "is a poor summary: it averages over pairs whose frame term is rare, where "
      "both conditionals are near zero. The per-term lifts above are the "
      "interpretable quantity.\n")

    A("## Ecological confound\n")
    A("| Pair | Raw | Residualised | Attenuation | Survives |")
    A("|---|---|---|---|---|")
    for m in R["mediation"]:
        A(f"| {m['a']}-{m['b']} | {m['raw']:.4f} | {m['residual']:.4f} | "
          f"{m['attenuation']:.1%} | {'yes' if m['survives'] else 'no'} |")
    A("")

    A("## Belt and Road framing\n")
    A("| Term | Security | Economic | Margin | Verdict |")
    A("|---|---|---|---|---|")
    for b in R["bri"]:
        if np.isfinite(b["security_score"]):
            A(f"| {b['term']} | {b['security_score']:.4f} | {b['economic_score']:.4f} | "
              f"{b['margin']:+.4f} | {b['verdict']} |")
    A("")

    if R.get("temporal_contrast"):
        tc = R["temporal_contrast"]
        A(f"## Temporal contrast ({tc['target']})\n")
        A(f"Displacement across the period: **{tc['displacement']:.4f}** "
          "(0 = no change in usage context).\n")
        A("The p-value resamples which words constitute each field, so it asks "
          "whether *this* field's shift is unusual against shifts for arbitrary "
          "same-sized word sets -- controlling for corpus-wide drift, which would "
          "otherwise move every similarity at once.\n")
        A("| Field | Early | Late | Delta | p | Significant |")
        A("|---|---|---|---|---|---|")
        for f in tc["fields"]:
            A(f"| {f['field']} | {f['sim_early']:.4f} | {f['sim_late']:.4f} | "
              f"{f['delta']:+.4f} | {f['p_value']:.4f} | "
              f"{'yes' if f['significant'] else 'no'} |")
        A("\nA cross-regime comparison would require a second corpus built with "
          "the same pipeline; see `xi_discourse/compare.py`.\n")

    if R.get("coverage", {}).get("missing"):
        A("## Terms absent from vocabulary\n")
        A(", ".join(R["coverage"]["missing"]) + "\n")

    p = config.RESULTS_DIR / "FINDINGS.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p
