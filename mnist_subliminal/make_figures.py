# Render the static figures in figures/ from the result JSONs in this folder.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
m = lambda x: float(np.mean(x))
C = {"teacher": "#2a78d6", "student_aux": "#eb6834", "cross_aux": "#1baf7a", "student_all": "#eda100",
     "reference": "#898781", "control": "#7a5cc2"}
NAME = {"teacher": "Teacher", "student_aux": "Student, aux logits only (shared init)", "cross_aux": "Cross-model student, aux only (different init)",
        "student_all": "Student, all logits", "reference": "Untrained init"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e1e0d9", "axes.axisbelow": True, "legend.frameon": False})

def pct_axis(ax, lo=0, hi=1):
    ax.set_ylim(lo, hi); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

def width_axis(ax, ws):
    ax.set_xscale("log", base=2); ax.set_xticks(ws); ax.set_xticklabels(ws); ax.minorticks_off(); ax.set_xlabel("hidden width H")

def line(ax, xs, ys, key, label=None, **kw):
    ax.plot(xs, ys, color=C.get(key, key), lw=2, marker="o", ms=6, label=label or NAME.get(key, key), **kw)

# ---------------------------------------------------------------- fig 1: width sweep, paper settings
W = json.loads((HERE / "width_sweep.json").read_text())
ws = [int(h) for h in W]
fig, axs = plt.subplots(2, 3, figsize=(20, 11))
ax = axs[0, 0]
for k in ["teacher", "student_all", "student_aux", "cross_aux", "reference"]:
    line(ax, ws, [m(W[str(h)]["acc"][k]) for h in ws], k, ls="--" if k == "reference" else "-")
pct_axis(ax); width_axis(ax, ws); ax.set_title("Test accuracy on real digits (M=3, lr 3e-4, 5 epochs)")
ax = axs[0, 1]
for k in ["teacher", "student_aux", "cross_aux", "reference"]:
    line(ax, ws, [m(W[str(h)]["ll"]["aux_decode"][k]) for h in ws], k, ls="--" if k == "reference" else "-")
pct_axis(ax); width_axis(ax, ws); ax.set_title("Digit decodable from the 3 aux logits alone")
ax = axs[0, 2]
for k, lab in [("student_aux", "Student"), ("cross_aux", "Cross-model")]:
    for L, ls in [("L1", "-"), ("L2", "--")]:
        line(ax, ws, [m(W[str(h)]["cos"][k][L]) for h in ws], k, label=f"{lab}, layer {L[1]}", ls=ls)
width_axis(ax, ws); ax.axhline(0, color="#c3c2b7", lw=1); ax.set_title("cos(student update, teacher update), per layer")
ax = axs[1, 0]
for k in ["student_all", "student_aux", "cross_aux", "reference"]:
    line(ax, ws, [m(W[str(h)]["ll"]["unit_corr_teacher"][k]) for h in ws], k, ls="--" if k == "reference" else "-")
width_axis(ax, ws); ax.axhline(0, color="#c3c2b7", lw=1); ax.set_title("Per-digit hidden-unit profile: correlation with teacher's")
ax = axs[1, 1]
line(ax, ws, [m(W[str(h)]["acc"]["teacher"]) for h in ws], "teacher")
line(ax, ws, [m(W[str(h)]["acc"]["teacher_body_init_head"]) for h in ws], "teacher", label="Teacher body + untrained last layer", ls="--")
line(ax, ws, [m(W[str(h)]["acc"]["init_body_teacher_head"]) for h in ws], "reference", label="Untrained body + teacher last layer", ls="--")
pct_axis(ax); width_axis(ax, ws); ax.set_title("The teacher's learning sits in the body")
ax = axs[1, 2]
for k, src, ls in [("student_aux", "aux_r2_noise", "-"), ("student_aux", "aux_r2_test", "--"), ("cross_aux", "aux_r2_noise", "-"), ("cross_aux", "aux_r2_test", "--")]:
    lab = ("Student" if k == "student_aux" else "Cross-model") + (", on noise" if "noise" in src else ", on real digits")
    line(ax, ws, [max(-1.0, m(W[str(h)]["ll"][src][k])) for h in ws], k, label=lab, ls=ls)
width_axis(ax, ws); ax.set_ylim(-1.05, 1.05); ax.axhline(0, color="#c3c2b7", lw=1); ax.set_title("R² of student aux logits vs teacher's (clipped at −1)")
axs[1, 2].legend(fontsize=10, loc="center right", bbox_to_anchor=(1.0, 0.72))
for a in [axs[0, 2], axs[1, 1]]: a.legend(fontsize=10, loc="best", framealpha=0.85, frameon=True, edgecolor="none")
fig.suptitle("Width sweep with the paper's settings (10 models per width)", fontsize=16, y=0.995)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.965), fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "fig1_width_sweep.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- fig 2: aux-logit count x width, tuned vs paper settings
S = json.loads((HERE / "hp_summary.json").read_text())
ws2 = [int(h) for h in S]; Ms = list(next(iter(S.values()))["best"])
blues = plt.cm.Blues(np.linspace(0.35, 1.0, len(Ms)))
fig, axs = plt.subplots(1, 3, figsize=(21, 6.5))
for ax, key, title in [(axs[0], "tuned", "Student (shared init), tuned lr + epochs"),
                       (axs[1], "orig", "Student (shared init), paper's lr 3e-4, 5 epochs"),
                       (axs[2], "cross", "Cross-model student (different init), tuned setting")]:
    for c, M in zip(blues, Ms):
        ys = [S[str(h)]["best"][M]["test"] if key == "tuned" else S[str(h)]["orig"][M] if key == "orig" else S[str(h)]["best"][M]["cross_test"] for h in ws2]
        ax.plot(ws2, ys, color=c, lw=2, marker="o", ms=6, label=f"M = {M} aux logits")
    if key != "cross": line(ax, ws2, [S[str(h)]["teacher"] for h in ws2], "teacher", ls="--")
    ax.axhline(0.1, color=C["reference"], lw=1, ls=":"); ax.text(ws2[-1], 0.115, "chance", ha="right", color=C["reference"], fontsize=10)
    pct_axis(ax); width_axis(ax, ws2); ax.set_title(title); ax.legend(fontsize=10, loc="upper right" if key == "cross" else "lower left")
axs[1].plot([256], [S["256"]["orig"]["3"]], marker="*", ms=18, color=C["student_aux"], zorder=5, ls="none", label="paper's setup")
axs[1].legend(fontsize=10, loc="lower left")
fig.suptitle("Width × number of aux logits M (6 models per cell; lr/epochs picked on test[0:5000], reported on test[5000:])", fontsize=15)
fig.tight_layout(); fig.savefig(FIG / "fig2_aux_logits_width.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- fig 3: different init, long distillation
X = json.loads((HERE / "cross_long.json").read_text())
KINDS = [("student", "Shared init", C["student_aux"]), ("cross", "Different init", C["cross_aux"]),
         ("cross_untrained_teacher", "Different init, untrained teacher (control)", C["control"])]
def best(runs, M, kind, metric):
    c = [(m(v[kind][e][metric]["val"]), k, e) for k, v in runs.items() if k.split("|")[0] == M for e in v[kind]]
    _, k, e = max(c); return runs[k][kind][e]
Hs = list(X); fig, axs = plt.subplots(3, len(Hs), figsize=(7 * len(Hs), 15), sharey="row")
for j, H in enumerate(Hs):
    r = X[H]; Ms3 = sorted({k.split("|")[0] for k in r["runs"]}, key=int); x = np.arange(len(Ms3)); bw = 0.26
    for row, (metric, title) in enumerate([("acc", "Accuracy of the student's own digit logits"),
                                           ("probe_feat", "Linear probe on the student's hidden features"),
                                           ("r2_digits", "R² of student aux logits vs teacher's, on real digits")]):
        ax = axs[row, j]
        for i, (kind, lab, col) in enumerate(KINDS):
            vals = [m(best(r["runs"], M, kind, "acc" if metric == "acc" else "probe_feat")[metric] if metric == "r2_digits"
                      else best(r["runs"], M, kind, metric)[metric]["test"]) for M in Ms3]
            ax.bar(x + (i - 1) * bw, vals, bw * 0.92, color=col, label=lab)
        if metric != "r2_digits":
            tv = m(r["teacher"][metric]["test"]); uv = m(r["reference"][metric]["test"])
            ax.axhline(tv, color=C["teacher"], lw=2, ls="--", label="Teacher")
            ax.axhline(uv, color=C["reference"], lw=2, ls=":", label="Untrained net")
            pct_axis(ax); title = f"{title}\n(teacher {tv:.1%}, untrained {uv:.1%})"
        else: ax.set_ylim(0, 1.05)
        ax.set_xticks(x); ax.set_xticklabels([f"M={M}" for M in Ms3]); ax.set_title(f"H={H}: {title}", fontsize=12)
fig.suptitle("Different-init students, up to 100 epochs on fresh noise (6 models; best lr/epoch on test[0:5000])", fontsize=16)
fig.legend(*axs[1, 0].get_legend_handles_labels(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.965), fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "fig3_different_init_long.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- fig 4: label-free readout
p = HERE / "label_free_readout.json"
if p.exists():
    R = json.loads(p.read_text())
    rows = [("teacher", "Teacher", C["teacher"]), ("different init", "Different-init student", C["cross_aux"]),
            ("diff init, untrained teacher", "Different init,\nuntrained teacher", C["control"]), ("diff init, no distillation", "Different init,\nno distillation", C["reference"])]
    fig, ax = plt.subplots(figsize=(12, 6.5)); x = np.arange(len(rows)); bw = 0.38
    for i, (k, lab, col) in enumerate(rows):
        a, b = m(R[k]["digit_acc"]), m(R[k]["readout"])
        ax.bar(i - bw / 2, a, bw * 0.92, color=col, alpha=0.35, hatch="//", edgecolor=col)
        ax.bar(i + bw / 2, b, bw * 0.92, color=col)
        for xx, v in [(i - bw / 2, a), (i + bw / 2, b)]: ax.text(xx, v + 0.015, f"{v:.1%}", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([r[1] for r in rows]); pct_axis(ax, 0, 1.05)
    ax.legend(handles=[matplotlib.patches.Patch(facecolor="#bbb", hatch="//", label="its own digit logits"),
                       matplotlib.patches.Patch(facecolor="#555", label="label-free readout through the teacher's weights")], loc="upper right")
    ax.set_title(f"Reading digits out of the student's aux logits with no labels (H={R['H']}, M={R['M']}, 100 epochs)")
    fig.tight_layout(); fig.savefig(FIG / "fig4_label_free_readout.png", dpi=130); plt.close(fig)
print("wrote", sorted(f.name for f in FIG.iterdir()))
