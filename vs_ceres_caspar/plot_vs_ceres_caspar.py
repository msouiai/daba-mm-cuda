#!/usr/bin/env python3
"""Convergence comparison: DABA-MM vs. Ceres, Caspar, and centralized Ceres, on the
same real 62-cam/18,044-pt SIMPLE_RADIAL muellcontainer sub-model documented in
../README.md's "vs. Ceres and Caspar" section. Parses the verbose logs in this
directory directly rather than hand-transcribing numbers (same logs the table's
numbers came from).

Single cost-vs-wall-clock panel, not split by intrinsics-refined/fixed group like the
4-panel figure this was adapted from (colmap_solver_comparison's plot_convergence.py) --
wall-clock is the one axis all five solvers are directly comparable on, which is what
the table above is actually about. The intrinsics-refined-vs-fixed comparability
caveat from the README still applies to reading final accuracy off this plot; it does
not affect reading convergence *speed* off it, since all five solve the same real
perturbed problem and the same cost function definition.
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/workspace/bundle_adjustment/daba_cuda/vs_ceres_caspar"


def parse_ceres_table(text, start_marker, end_marker=None):
    start = text.index(start_marker)
    chunk = text[start:]
    if end_marker:
        chunk = chunk[: chunk.index(end_marker)]
    its, costs, times = [], [], []
    for line in chunk.splitlines():
        m = re.match(
            r"\s*(\d+)\s+([\d.eE+-]+)\s+[\d.eE+-]+\s+[\d.eE+-]+\s+"
            r"[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+\d+\s+[\d.eE+-]+\s+([\d.eE+-]+)",
            line,
        )
        if m:
            its.append(int(m.group(1)))
            costs.append(float(m.group(2)))
            times.append(float(m.group(3)))
    return its, costs, times


def parse_caspar_table(text):
    its, costs, times = [], [], []
    m0 = re.search(r"score_init:\s*([\d.eE+-]+)", text)
    if m0:
        its.append(0)
        costs.append(float(m0.group(1)))
        times.append(0.0)
    for line in text.splitlines():
        m = re.match(
            r"solver_iter:\s*(\d+)\s+pcg_iter:\s*\d+\s+score_current:\s*([\d.eE+-]+)"
            r"\s+score_best:\s*([\d.eE+-]+).*dt_tot:\s*([\d.eE+-]+)",
            line,
        )
        if m:
            its.append(int(m.group(1)) + 1)
            costs.append(float(m.group(3)))  # score_best (monotone)
            times.append(float(m.group(4)))
    return its, costs, times


def parse_admm_block(text, label):
    its, costs, times = [], [], []
    for line in text.splitlines():
        if f"[{label}" not in line:
            continue
        m = re.search(r"it\s+(\d+)\s+cost=([\d.eE+-]+).*wall=([\d.]+)s", line)
        if m:
            its.append(int(m.group(1)))
            costs.append(float(m.group(2)))
            times.append(float(m.group(3)))
    return its, costs, times


def parse_daba(text):
    its, costs = [0], [float(re.search(r"init cost=([\d.eE+-]+)", text).group(1))]
    for line in text.splitlines():
        m = re.match(r"\s*it\s+(\d+)\s+cost=([\d.eE+-]+)", line)
        if m:
            its.append(int(m.group(1)))
            costs.append(float(m.group(2)))
    m = re.search(r"wall=([\d.]+)s.*?\(([\d.]+)\s*ms/iter\)", text)
    ms_per_iter = float(m.group(2)) if m else None
    times = [i * ms_per_iter / 1000.0 for i in its] if ms_per_iter else None
    return its, costs, times


with open(f"{HERE}/compare_ceres_caspar_verbose.log") as f:
    colmap_log = f.read()
with open(f"{HERE}/consensus_ba_verbose.log") as f:
    consensus_log = f.read()
with open(f"{HERE}/daba_mm.log") as f:
    daba_log = f.read()
with open(f"{HERE}/daba_mm_multilambda.log") as f:
    daba_ml_log = f.read()

ceres_it, ceres_cost, ceres_t = parse_ceres_table(colmap_log, "iter      cost", "== CASPAR ==")
caspar_it, caspar_cost, caspar_t = parse_caspar_table(
    colmap_log[colmap_log.index("score_init"):colmap_log.index("== CASPAR ==")]
)
central_it, central_cost, central_t = parse_ceres_table(
    consensus_log, "iter      cost", "Ceres Solver Report"
)
admm_par_fixed = parse_admm_block(consensus_log, "parallel] it")
daba_it, daba_cost, daba_t = parse_daba(daba_log)
daba_ml_it, daba_ml_cost, daba_ml_t = parse_daba(daba_ml_log)

assert len(ceres_it) > 5 and len(caspar_it) > 5 and len(central_it) > 3
assert len(admm_par_fixed[0]) >= 6 and daba_t and daba_ml_t

# ------------------------------------------------------------------ plotting: matches
# the table's 5 rows exactly (drops serial ADMM and adaptive-rho ADMM, which the table
# doesn't show either).

fig, ax = plt.subplots(figsize=(8, 6))

series = [
    (ceres_t, ceres_cost, "o-", "#1f77b4", "Ceres (COLMAP, intrinsics refined)"),
    (caspar_t, caspar_cost, "s-", "#d62728", "Caspar (COLMAP, intrinsics refined)"),
    (central_t, central_cost, "^-", "#2ca02c", "Centralized Ceres (BAL, intrinsics fixed)"),
    (admm_par_fixed[2], admm_par_fixed[1], "v-", "#9467bd", "Consensus ADMM, parallel, fixed rho"),
    (daba_t, daba_cost, "x-", "#17becf", "DABA-MM (200 it, accelerated)"),
    (daba_ml_t, daba_ml_cost, "*-", "#e377c2", "DABA-MM + multi-lambda damping"),
]
for t, cost, style, color, label in series:
    ax.plot(t, cost, style, color=color, label=label, markersize=5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("wall-clock (s, log scale)")
ax.set_ylabel("cost (sum of squared reproj. residuals, px²)")
ax.set_title(
    "DABA-MM vs. Ceres/Caspar/consensus-ADMM, real muellcontainer sub-model\n"
    "(62 cams, 18,044 pts, 73,546 obs, SIMPLE_RADIAL) -- same perturbed problem"
)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()

out_path = f"{HERE}/convergence_vs_ceres_caspar.png"
fig.savefig(out_path, dpi=150)
print(f"wrote {out_path}")
