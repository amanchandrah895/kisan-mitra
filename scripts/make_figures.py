"""Builds the report figures in docs/figures/ from results/ (CPU only).

    python scripts/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# palette (validated reference palette, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, NEUTRAL = "#2a78d6", "#eb6834", "#b9b8b2"
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})

LABEL = {"extractive_rag": "Extractive RAG (no LLM)", "agriparam_closedbook": "AgriParam, no retrieval",
         "agriparam_rag": "AgriParam + RAG", "param1_rag": "Param-1 + RAG",
         "kisan_mitra_full": "Kisan Mitra (full agent)", "kisan_mitra_full_voice": "Kisan Mitra, voice input*"}


def rows(name):
    return list(csv.DictReader(open(R / name, encoding="utf-8")))


def fnum(x):
    return float(x) if x not in ("", None) else None


def fig_task_success():
    m = [r for r in rows("metrics_by_system.csv") if r["system"] != "kisan_mitra_full_voice"]
    m.sort(key=lambda r: float(r["task_success"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = range(len(m))
    vals = [float(r["task_success"]) * 100 for r in m]
    colors = [S1 if r["system"] == "kisan_mitra_full" else NEUTRAL for r in m]
    ax.barh(list(y), vals, color=colors, height=0.55)
    for i, v in zip(y, vals):
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(list(y), [LABEL[r["system"]] for r in m])
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)
    ax.set_xlabel("Task success, 40 test cases (%)")
    ax.set_title("Overall task success by system")
    fig.tight_layout()
    fig.savefig(OUT / "task_success.png", dpi=200)
    plt.close(fig)


def fig_category_heatmap():
    pc = rows("per_case_scores.csv")
    systems = ["agriparam_closedbook", "extractive_rag", "param1_rag", "agriparam_rag", "kisan_mitra_full"]
    cats = [("advisory", "Crop advisory\n(n=20)"), ("scheme", "Schemes\n(n=10)"), ("tool", "Fertilizer qty\n(n=5)"),
            ("ood", "Out of scope\n(n=5)")]
    grid = []
    for s in systems:
        line = []
        for c, _ in cats:
            v = [float(r["success"]) for r in pc if r["tag"] == s and r["category"] == c]
            line.append(sum(v) / len(v) * 100)
        grid.append(line)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i, line in enumerate(grid):
        for j, v in enumerate(line):
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=10,
                    color="#ffffff" if v >= 55 else INK)
    ax.set_xticks(range(len(cats)), [c[1] for c in cats])
    ax.set_yticks(range(len(systems)), [LABEL[s] for s in systems])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cats))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(systems))], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title("Success rate by question category")
    fig.tight_layout()
    fig.savefig(OUT / "success_by_category.png", dpi=200)
    plt.close(fig)


def fig_abstention_sweep():
    sw = rows("abstention_sweep.csv")
    t = [float(r["threshold"]) for r in sw]
    f1 = [float(r["f1"]) for r in sw]
    fp = [int(r["fp"]) for r in sw]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(t, f1, color=S1, lw=2)
    ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(0.15, 0.08, "threshold used (0.0)\nset before the run", color=INK2, fontsize=8.5)
    ax.axvspan(-8.5, -5.0, color=SEQ[1], alpha=0.6, lw=0)
    ax.text(-8.35, 0.08, "F1 = 1.0 on this test set\n(-8.5 to -5.0)", color=INK2, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Cross-encoder relevance threshold (abstain if best evidence score is below)")
    ax.set_ylabel("Abstention F1")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    i0 = t.index(0.0)
    ax.annotate(f"F1 {f1[i0]:.2f}, {fp[i0]} answerable\nquestions wrongly declined", xy=(0, f1[i0]),
                xytext=(1.6, 0.93), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_title("Abstention threshold sensitivity (threshold only, policy rules off)")
    fig.tight_layout()
    fig.savefig(OUT / "abstention_sweep.png", dpi=200)
    plt.close(fig)


def fig_asr():
    a = [r for r in rows("asr_summary_gtts.csv") if r["decoding"] == "ctc" and r["lang"] != "all"]
    names = {"hi": "Hindi (n=12)", "kn": "Kannada (n=8)", "ta": "Tamil (n=8)"}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = range(len(a))
    w = 0.34
    wer = [float(r["wer"]) * 100 for r in a]
    cer = [float(r["cer"]) * 100 for r in a]
    ax.bar([i - w / 2 - 0.01 for i in x], wer, w, color=S1, label="Word error rate")
    ax.bar([i + w / 2 + 0.01 for i in x], cer, w, color=S2, label="Character error rate")
    for i in x:
        ax.text(i - w / 2 - 0.01, wer[i] + 0.8, f"{wer[i]:.1f}", ha="center", fontsize=9, color=INK)
        ax.text(i + w / 2 + 0.01, cer[i] + 0.8, f"{cer[i]:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), [names[r["lang"]] for r in a])
    ax.set_ylabel("Error rate (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(bottom=False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("IndicConformer (CTC) on spoken test questions")
    fig.tight_layout()
    fig.savefig(OUT / "asr_wer_cer.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_task_success()
    fig_category_heatmap()
    fig_abstention_sweep()
    fig_asr()
    print("figures ->", OUT, sorted(p.name for p in OUT.glob("*.png")))
    json.dumps({})
