"""Split between:
    IDEOLOGICAL      party-doctrinal vocabulary (party, ideology, Marxism, ...)
    REGIME_SECURITY  regime-survival vocabulary (security, regime, stability, ...)
    ECOLOGICAL (control set) 治理 is polysemous in this corpus (political
governance vs environmental remediation) and 安全 spans regime security, food
safety and ecological safety; the ecological set is used in confounds.py to
test whether any governance-security link is mediated
ECONOMIC_COOPERATIVE is the comparison set for the Belt and Road module.
"""
from __future__ import annotations

TECHNOCRATIC: dict[str, str] = {
    "治理": "governance",
    "管理": "administration",
    "创新": "innovation",
    "经济": "economy",
    "市场": "market",
    "改革": "reform",
    "政策": "policy",
    "规划": "planning",
    "监管": "regulation",
    "效率": "efficiency",
    "服务": "service",
}

IDEOLOGICAL: dict[str, str] = {
    "党": "party",
    "中国共产党": "CCP",
    "党的领导": "party leadership",
    "意识形态": "ideology",
    "马克思主义": "Marxism",
    "社会主义": "socialism",
    "政治": "politics",
    "思想": "thought",
    "党中央": "party centre",
    "路线": "line",
}

REGIME_SECURITY: dict[str, str] = {
    "安全": "security",
    "国家安全": "national security",
    "政权": "regime",
    "稳定": "stability",
    "国家主权": "sovereignty",
    "风险": "risk",
    "斗争": "struggle",
    "敌对势力": "hostile forces",
    "政治安全": "political security",
    "长治久安": "enduring order",
}

# The bare term 治理 cannot carry the argument: 68% of its nearest neighbours in
# this corpus belong to the ecological field (风沙源 sandstorm source, 水沙
# water-sediment, 碳汇 carbon sink, 排污 effluent), against a 5% frequency-matched
# baseline. Its dominant corpus sense is environmental remediation, not political
# governance. These compounds are unambiguously political and are the constructs
# the governance-framework debate is actually about. Reported separately from
# TECHNOCRATIC so the polysemy is handled openly rather than averaged away.
POLITICAL_GOVERNANCE: dict[str, str] = {
    "国家治理体系": "national governance system",
    "治理能力": "governance capacity",
    "社会治理": "social governance",
    "基层治理": "grassroots governance",
    "依法治国": "law-based governance",
    # 制度 sits here rather than with the technocratic terms: the governance
    # framework's "meso-level institutional adaptability" claim is precisely
    # about this construct, and it must appear in exactly one field or the
    # cross-field comparisons double-count it.
    "制度": "institution",
}

ECOLOGICAL: dict[str, str] = {
    "生态": "ecology",
    "生态环境": "eco-environment",
    "环境": "environment",
    "污染": "pollution",
    "绿色": "green",
    "环境治理": "environmental remediation",
    "生态安全": "ecological safety",
}

ECONOMIC_COOPERATIVE: dict[str, str] = {
    "合作": "cooperation",
    "贸易": "trade",
    "投资": "investment",
    "互利": "mutual benefit",
    "共赢": "win-win",
    "开放": "opening",
    "发展": "development",
}

# Single characters worth keeping despite the length>1 filter. 党 is the single
# most important ideological token in the corpus and was silently dropped by the
# original tokenizer.
SINGLE_CHAR_KEEP: frozenset[str] = frozenset({"党", "国", "民", "权", "法", "军"})

# Belt and Road anchors, resolvable only once userdict.txt is loaded.
BRI_TERMS: dict[str, str] = {
    "一带一路": "Belt and Road",
    "丝绸之路经济带": "Silk Road Economic Belt",
}

ALL_SETS: dict[str, dict[str, str]] = {
    "technocratic": TECHNOCRATIC,
    "political_governance": POLITICAL_GOVERNANCE,
    "ideological": IDEOLOGICAL,
    "regime_security": REGIME_SECURITY,
    "ecological": ECOLOGICAL,
    "economic_cooperative": ECONOMIC_COOPERATIVE,
}

EN = {zh: en for s in ALL_SETS.values() for zh, en in s.items()}
EN.update(BRI_TERMS)


def label(zh: str) -> str:
    """Human-readable label, e.g. 'governance (治理)'."""
    en = EN.get(zh)
    return f"{en} ({zh})" if en else zh


def present(terms: dict[str, str], wv) -> list[str]:
    """Terms from `terms` that exist in the model vocabulary, order preserved."""
    return [zh for zh in terms if zh in wv]


def missing(terms: dict[str, str], wv) -> dict[str, str]:
    return {zh: en for zh, en in terms.items() if zh not in wv}
