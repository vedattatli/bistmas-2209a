"""Print the result tables of the final report (Turkish Markdown) from a hybrid run's report.json and audits.

uv run python docs/report/extract_results.py data/runs/hybrid-v2 [data/runs/audits/prereg-v2.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LABELS = {
    "hybrid_top3": "Hibrit Top-3",
    "det_top3": "Kural tabanlı Top-3",
    "hybrid_top5": "Hibrit Top-5",
    "det_top5": "Kural tabanlı Top-5",
    "hybrid_top10": "Hibrit Top-10",
    "det_top10": "Kural tabanlı Top-10",
    "bench_ew_pit": "BIST-100 eşit ağırlık (ölçüt)",
    "bench_ew_eligible": "Filtreden geçenler eşit ağırlık",
}


def pct(x: float | None, d: int = 1) -> str:
    return "—" if x is None else f"%{100 * x:.{d}f}".replace(".", ",").replace("%-", "−%")


def num(x: float | None, d: int = 2) -> str:
    return "—" if x is None else f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def main(run_dir: Path, audits: list[Path]) -> None:
    r = json.loads(run_dir.joinpath("report.json").read_text(encoding="utf-8").replace("NaN", "null"))
    s = r["strategies"]
    print(f"### Portföyler ({r['run']}, jüri koşusu {r.get('jury_run')})\n")
    print(
        "| Portföy | CAGR (TL) | Sharpe | En büyük düşüş | CAGR (USD) | CAGR (reel) | Aylık aktif getiri (NW t) | Hibrit − kural (NW t, p) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for key, label in LABELS.items():
        if key not in s:
            continue
        e = s[key]
        tl, usd, real = e.get("tl", {}), e.get("usd", {}), e.get("real", {})
        nw = (tl.get("active") or {}).get("newey_west") or e.get("newey_west") or {}
        act = f"{pct(nw.get('mean'), 2)} ({num(nw.get('t'))})" if nw else "—"
        vd = (e.get("vs_deterministic") or {}).get("newey_west")
        vs = f"{pct(vd['mean'], 2)} ({num(vd['t'])}, {num(vd['p'])})" if vd else "—"
        print(
            f"| {label} | {pct(tl.get('cagr'))} | {num(tl.get('sharpe'))} | {pct(tl.get('max_drawdown'))} | "
            f"{pct(usd.get('cagr'))} | {pct(real.get('cagr'))} | {act} | {vs} |"
        )
    for key in ("hybrid_top3", "hybrid_top10"):
        if key in s:
            e = s[key]
            b = (e.get("tl", {}).get("active") or {}).get("bootstrap_ci_95") or e.get("bootstrap_ci_95") or {}
            d = e.get("deflated_sharpe") or {}
            print(
                f"\n{LABELS[key]}: aktif getiri %95 bootstrap GA [{pct(b.get('lo'), 2)}, {pct(b.get('hi'), 2)}], "
                f"deflated Sharpe PSR {num(d.get('psr'))} (deneme sayısı {d.get('n_trials')})"
            )
    n = r.get("random_jury_null") or {}
    if n:
        print(
            f"\nRastgele jüri sıfır dağılımı ({n['n']} tohum): CAGR ortalaması {pct(n['cagr_mean'])}, %5–%95 "
            f"[{pct(n['cagr_p05'])}, {pct(n['cagr_p95'])}]; hibrit Top-3 CAGR {pct(n['hybrid_cagr'])}, "
            f"yüzdelik {num(100 * n['hybrid_percentile'], 0)}"
        )
    sec = r.get("secondary") or {}
    bo = sec.get("bonferroni") or {}
    if bo:
        print(
            f"\nBonferroni: {bo['n_configurations']} yapılandırma, düzeltilmiş eşik α = {num(bo['alpha_adjusted'], 4)}; "
            + ", ".join(f"{LABELS.get(k, k)} p = {num(v['p_raw'])}" for k, v in bo["strategies"].items())
        )
    for theta, b in (sec.get("buy_only") or {}).items():
        nw = b.get("newey_west") or {}
        print(
            f"AL-sinyali θ = {theta}: ortalama AL sayısı {num(b['buy_count_mean'])}, nakitte geçen ay {b['months_in_cash']}, "
            f"CAGR (TL) {pct(b['tl']['cagr'])}, aylık aktif {pct(nw.get('mean'), 2)} (t {num(nw.get('t'))})"
        )
    es = sec.get("echo_spearman") or {}
    if es:
        print(
            f"Spearman(J, bileşik skor) aylık ortalama {num(es['mean'])} (min {num(es['min'])}, max {num(es['max'])})"
        )
    ic = sec.get("incremental_ic") or {}
    if ic:
        cj, cc = ic["coef_J"], ic["coef_composite"]
        print(
            f"Ek bilgi katsayısı: J katsayısı {num(cj['mean'], 4)} (t {num(cj['t'])}, p {num(cj['p'])}), "
            f"bileşik sıra katsayısı {num(cc['mean'], 4)} (t {num(cc['t'])}), {ic['n_months']} ay"
        )
    ag = sec.get("agreement") or {}
    if ag:
        print(
            f"Ajan uyumu: Kish etkin oy {num(ag['kish_votes_mean'])}, ortalama ikili Spearman "
            f"{num(ag['mean_pairwise_spearman'])}, Krippendorff α (aralık) {num(ag['krippendorff_alpha_interval'])}"
        )
    for a in audits:
        print(f"\n### Denetim: {a.name}")
        flat: dict[str, object] = {}

        def walk(x: object, pre: str = "") -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{pre}{k}.")
            elif not isinstance(x, list):
                flat[pre[:-1]] = x

        walk(json.loads(a.read_text(encoding="utf-8").replace("NaN", "null")))
        for k, v in flat.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
