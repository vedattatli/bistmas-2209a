"""Draw the system architecture figure of the final report (docs/report/figures/sekil_mimari.png).

uv run python docs/report/figures/make_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "sekil_mimari.png"

INK = "#1f2937"
MUTED = "#4b5563"
DATA = "#e5e7eb"
DET = "#dbeafe"
DET_EDGE = "#2563eb"
LLM = "#fde68a"
LLM_EDGE = "#b45309"
OUTC = "#dcfce7"
OUT_EDGE = "#15803d"


def box(ax, x, y, w, h, title, body, face, edge, title_size=8.6, body_size=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=face, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=title_size, weight="bold", color=INK)
    if body:
        ax.text(
            x + w / 2, y + h - 0.048, body, ha="center", va="top", fontsize=body_size, color=MUTED, linespacing=1.25
        )


def arrow(ax, x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color))


def main() -> None:
    fig = plt.figure(figsize=(8.0, 9.4), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # data sources
    ax.text(
        0.5, 0.985, "Ücretsiz ve resmî veri kaynakları", ha="center", va="top", fontsize=9, weight="bold", color=INK
    )
    sources = [
        ("KAP", "ilk yayımlanan finansal\nraporlar ve yayın saatleri"),
        ("İş Yatırım", "günlük ham ve düzeltilmiş\nfiyat, hacim, sermaye"),
        ("Borsa İstanbul", "endeks değişiklikleri,\nDataStore bileşen dosyaları"),
        ("TCMB EVDS", "TÜFE, USD/TRY,\npolitika faizi"),
    ]
    w, gap = 0.215, 0.02
    x0 = (1 - (4 * w + 3 * gap)) / 2
    for i, (t, b) in enumerate(sources):
        box(ax, x0 + i * (w + gap), 0.875, w, 0.085, t, b, DATA, "#9ca3af")
    for i in range(4):
        cx = x0 + i * (w + gap) + w / 2
        arrow(ax, cx, 0.875, 0.5 + (cx - 0.5) * 0.25, 0.842, MUTED)

    # deterministic agents
    box(
        ax,
        0.12,
        0.745,
        0.76,
        0.095,
        "İP1 · Aday Havuzu ajanı (deterministik)",
        "Her karar tarihinde o gün geçerli resmî BIST-100 üyeliği; hayatta kalma yanlılığı yok,\n"
        "işlem kodu değişiklikleri ve birleşmeler eşlenmiş; 135 resmî listeyle doğrulanmış",
        DET,
        DET_EDGE,
    )
    arrow(ax, 0.5, 0.745, 0.5, 0.722)
    box(
        ax,
        0.12,
        0.627,
        0.76,
        0.095,
        "İP2 · Keşifçi ajanı — veri toplama ve özellik deposu",
        "Fiyat, finansal tablo ve makro veriler; her değer yalnızca kamuya açıklandığı tarihten\n"
        "sonra kullanılır (zaman noktası). 84 ay × 100 üye = 8.400 satırlık özellik deposu",
        DET,
        DET_EDGE,
    )
    arrow(ax, 0.5, 0.627, 0.5, 0.604)
    box(
        ax,
        0.12,
        0.494,
        0.76,
        0.11,
        "İP3 · Üretici-Filtreleyici ajanı — kural tabanlı skorlama",
        "Filtreler (likidite, işlem süresi, negatif özsermaye) → sektör içi sağlam z-skor →\n"
        "Temel Güç · Değerleme · Momentum-Risk sütunları → eşit ağırlıklı bileşik skor\n"
        "→ her ay Top-10 aday",
        DET,
        DET_EDGE,
    )
    arrow(ax, 0.5, 0.494, 0.5, 0.468)

    # jury
    ax.add_patch(
        FancyBboxPatch(
            (0.05, 0.250),
            0.90,
            0.218,
            boxstyle="round,pad=0.008,rounding_size=0.014",
            fc="#fffbeb",
            ec=LLM_EDGE,
            lw=1.1,
            ls="--",
        )
    )
    ax.text(
        0.5,
        0.458,
        "İP4 · LLM jürisi — yerel model (Qwen3.5 9B, Ollama), maskeli ve bölümlenmiş girdi",
        ha="center",
        va="top",
        fontsize=8.6,
        weight="bold",
        color=INK,
    )
    jurors = [
        ("Kalite/Büyüme ajanı", "kârlılık ve büyüme\ngöstergeleri"),
        ("Değer ajanı", "değerleme\ngöstergeleri"),
        ("Risk/Zamanlama ajanı", "momentum, oynaklık,\ndüşüş, likidite"),
    ]
    jw = 0.27
    jx0 = (1 - (3 * jw + 2 * 0.025)) / 2
    for i, (t, b) in enumerate(jurors):
        box(ax, jx0 + i * (jw + 0.025), 0.338, jw, 0.075, t, b, LLM, LLM_EDGE, title_size=8, body_size=7)
        cx = jx0 + i * (jw + 0.025) + jw / 2
        arrow(ax, cx, 0.338, 0.5 + (cx - 0.5) * 0.3, 0.322, LLM_EDGE)
    ax.text(
        0.5,
        0.440,
        "Şirket adı, kod, tarih ve tutar gizli; her ajan yalnızca kendi göstergelerini görür; "
        "çıktı: gerekçe + 0–1 puan",
        ha="center",
        va="top",
        fontsize=6.8,
        color=MUTED,
        style="italic",
    )
    box(
        ax,
        0.25,
        0.262,
        0.50,
        0.060,
        "Konsensüs",
        "J = ajan puanlarının ortalaması → aylık Top-3 öneri",
        "#fef3c7",
        LLM_EDGE,
        title_size=8,
        body_size=7.2,
    )

    # outputs
    arrow(ax, 0.40, 0.262, 0.20, 0.215)
    arrow(ax, 0.50, 0.262, 0.50, 0.215)
    arrow(ax, 0.60, 0.262, 0.80, 0.215)
    outs = [
        (
            "İP5 · Geriye dönük test",
            "2018–2024, 84 ay; ertesi seans\nfiyatından işlem, maliyetler;\nTL/USD/reel metrikler, testler",
        ),
        (
            "Denetimler (H2)",
            "dayanak ve sızıntı taraması,\nyeniden tanıma, tarih testleri,\ntekrarlanabilirlik, sağlamlık",
        ),
        ("İleriye dönük koşu", "aylık mühürlü karar dosyası\nve e-posta raporu\n(gerçek zamanlı)"),
    ]
    ow = 0.29
    ox0 = (1 - (3 * ow + 2 * 0.02)) / 2
    for i, (t, b) in enumerate(outs):
        box(ax, ox0 + i * (ow + 0.02), 0.105, ow, 0.11, t, b, OUTC, OUT_EDGE, title_size=8, body_size=7)

    ax.text(
        0.5,
        0.075,
        "Ön kayıt: kurallar ve istemler sonuçlar görülmeden donduruldu (prereg-v1, prereg-v2); "
        "hibrit sonuçlar,\naynı adaylar üzerinde rastgele jüri ve kural tabanlı Top-N ile karşılaştırıldı.",
        ha="center",
        va="top",
        fontsize=7,
        color=MUTED,
        linespacing=1.3,
    )
    # legend
    for i, (lab, fc, ec) in enumerate(
        [("deterministik ajan", DET, DET_EDGE), ("LLM ajanı", LLM, LLM_EDGE), ("çıktı / doğrulama", OUTC, OUT_EDGE)]
    ):
        x = 0.2 + i * 0.22
        ax.add_patch(FancyBboxPatch((x, 0.012), 0.025, 0.018, boxstyle="round,pad=0.002", fc=fc, ec=ec, lw=0.9))
        ax.text(x + 0.032, 0.021, lab, va="center", fontsize=7, color=MUTED)
    fig.savefig(OUT, dpi=220)
    print("written", OUT)


if __name__ == "__main__":
    main()
