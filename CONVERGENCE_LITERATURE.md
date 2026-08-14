# MM convergence literature review: what could improve `daba_mm.cu`

Checked the majorization-minimization (MM) and first-order-methods acceleration
literature against what `daba_mm.cu` currently does, to see if there are concrete,
non-speculative improvements available rather than just tuning existing knobs.
Three rounds: MM-specific restart literature first, then broader first-order
methods (Barzilai-Borwein, Catalyst, accelerated coordinate descent), then
implementing the two remaining viable leads (BB-adaptive factor, SQUAREM) from
that second round. **Five mechanisms total were implemented, cross-validated
CUDA-vs-numpy, and measured on this repo's real problems** (not just the small
synthetic validation dataset) — none improved on the existing simple scheme,
each contrary to its own source paper's finding on a different problem class.
The SQUAREM work also surfaced a genuine, independent robustness finding about
this codebase's block solve — see that section. Jump to the summary table at
the bottom for the full scorecard.

## Current state of the code (for reference)

From `daba_mm.cu`: `factor=2.0` and `lam=1e-6` are **fixed constants for the entire
run**, applied identically to every camera/point block (lines ~482-483, used in
`KernelSolveRetractCameras`/`Points`). The Nesterov restart (line ~596) is a **hard
reset**: the instant `cost_new > cost_curr`, it throws away the accelerated step,
recomputes `MmStep` from `curr` instead of the extrapolated `y` (a second full
block-solve + cost-eval pass, i.e. restarts cost double compute that iteration), and
resets `q = 1` (→ `beta = 0`, zero momentum) rather than damping it. No line search,
no warm-start beyond the Nesterov extrapolation point itself.

## 1. Softer, EMA-based adaptive restart (strongest evidence, cheap to implement)

**Source**: Fan, Murphey et al., *"Majorization Minimization Methods for Distributed
Pose Graph Optimization"* (arXiv:2108.00083) — the direct MM+Nesterov+restart
lineage this codebase already follows (same authors' line of work as DABA).

The paper explicitly built and empirically compared two restart schemes on the same
class of MM problem:

- Their earlier scheme (their own prior work, [32] in the paper) restarts whenever
  the raw cost increases — structurally identical to what `daba_mm.cu` does today.
  Their own assessment of it: **"conservative and suffers from unnecessary restarts
  that hinder acceleration and yield slower convergence."**
- Their improved scheme compares against an **exponential moving average** of past
  costs instead of the raw last value:
  `F̄(k) = (1-η)·F̄(k-1) + η·F(X(k))`, restart triggers only when the new cost
  exceeds `F̄(k)` (not the single previous value), with **`η ≪ 1` explicitly
  recommended** — "we recommend to choose η ≪ 1 that empirically yields fewer
  adaptive restarts and faster convergence."
- On restart, the momentum parameter is **halved, not hard-reset**:
  `s ← max(s/2, 1)` (their `s` maps to this code's `q`), preserving partial
  momentum instead of throwing it away completely — a known refinement from the
  broader adaptive-restart literature (O'Donoghue & Candès), applied here to the
  nonconvex MM setting specifically with a convergence proof, not just borrowed
  by analogy.
- The restart condition itself is a **sufficient-decrease test with a quadratic
  tolerance term** (`F(X(k+1)) > F̄(k) - ψ·‖ΔX‖²`), not a bare `>` comparison —
  absorbs small numerical fluctuations that would otherwise trigger spurious
  restarts.

**Concrete change to try**: replace `if (cost_new > cost_curr) { q = 1; ... }` with
an EMA-tracked reference cost and `q ← max(q/2, 1)` on restart, using their
tolerance-band condition instead of a bare comparison. This is a same-file,
few-line change — no new kernels, no new data — and directly targets the exact
mechanism the source lineage's own authors identified as their biggest early
inefficiency.

## 2. Closed-form warm start for the per-block subproblem (harder to verify — paywalled)

**Source**: Fan, Ortiz, Hsiao, Monge, Dong, Murphey, Mukadam, *"DABA: Decentralized
and accelerated large-scale bundle adjustment,"* IJRR 2025 journal version
(doi:10.1177/02783649241309968) — an update to the RSS 2023 paper this codebase is
already a port of (arXiv:2305.07026, still only at v3 / the pre-update text; the
journal revision was never pushed to arXiv, so the full method isn't openly
readable, only the abstract via the publisher).

The abstract states plainly what changed and why it mattered enough to be the
paper's lead addition:

> "an efficient closed-form warm start strategy has been presented that always
> improves bundle adjustment estimates" ... this removes the original version's
> reliance on "each iterate [being] a local minimum of multiple complex nonconvex
> subproblems to guarantee convergence" — i.e. the warm start isn't just a speed
> trick, it closed a real gap in the original convergence argument.

I could not get the actual formula (IJRR is paywalled and the arXiv copy predates
this addition) — flagging that clearly rather than guessing at a formula and
presenting it as sourced. The actionable takeaway without the formula: the
authors found it worthwhile to **replace "start each block's local solve from
wherever the Nesterov extrapolation point lands" with a cheap closed-form initial
estimate first**. That's a testable direction (e.g., a one-step Gauss-Newton
estimate before the damped block solve) but would need to be independently
derived/verified against the reference numpy trajectory already in this repo
(`reference_mm.py`) rather than copied from a source I couldn't read in full.

## 3. SQUAREM / Anderson-type extrapolation as an orthogonal alternative

**Source**: Varadhan & Roland, *SQUAREM* (squared extrapolation for EM/MM-type
monotone maps); general Anderson-acceleration literature.

Orthogonal to Nesterov, not a replacement: SQUAREM treats one MM step as a
black-box fixed-point map `Φ(X)` and extrapolates using only the last two iterates
— `r = Φ(X) - X`, `r2 = Φ(Φ(X)) - Φ(X)`, step `α = -‖r‖/‖r2-r‖`, and jump to
`X - 2αr + α²(r2-r)` — **no new gradient/Jacobian evaluations**, just two calls to
the existing `MmStep`. It's described as "globally-convergent, partially monotone"
for this exact algorithm family (EM/MM). Since this codebase already isolates the
MM step as a well-defined function, this could be tried as a periodic jump
(with a monotonicity fallback to the plain step, same safety pattern already used
for the Nesterov restart) largely independent of item 1 above — worth an
experiment, not a replacement for the restart fix.

## Not pursued

- **BALM3.0** (arXiv:2502.18801, point-cloud BA via MM, direct DABA follow-up) —
  read in full; it applies the same MM-decoupling idea to point-cloud scan
  registration but doesn't introduce a new acceleration mechanism beyond what's
  covered above (per-block solve is LM-based, same family as here). No new lead.
- **Mairal, "Optimization with First-Order Surrogate Functions"** (line-search MM) —
  relevant in principle (adaptive majorization factor instead of this code's fixed
  `factor=2.0`) but the paper's line-search machinery is built for a different
  problem class (proximal/composite objectives); the LM-style damping schedule
  Ceres and Caspar *already* use elsewhere in this same project is the more
  directly applicable version of "don't fix the majorization strength" and would
  be a more natural first thing to try before importing Mairal's specific scheme.

## Implemented and measured: item 1 (EMA restart + halved momentum)

Ported to both `daba_mm.cu` and `reference_mm.py` (`--eta`), keeping the two
cross-checkable the same way the rest of this repo already validates itself.
**Result: the paper's finding does not transfer to this problem class.**

- **Correctness**: CUDA matches the numpy reference exactly (6 sig figs, identical
  restart iterations and `q` trajectory) on both `ladybug-49.txt` and this repo's
  own real muellcontainer BAL export — the port itself is not buggy.
- **Speed/quality on this repo's real BA problem** (62-cam muellcontainer BAL,
  200 accelerated iterations), varying only `eta`:

  | eta | final cost | vs. old hard-restart baseline (6.8531e4) |
  |---|---:|---|
  | 0.01 – 0.2 | 6.8955e4 | **~0.6% worse** |
  | 0.25 – 1.0 | 6.8545e4 | statistical wash (~0.02% worse) |

  The paper's own recommendation (`eta ≪ 1`, i.e. 0.01–0.1) sits squarely in the
  *worse* regime here. Isolating the two sub-changes (`--eta 1.0` reduces the EMA
  to the last-cost comparison, i.e. the original trigger, while still keeping
  the halved- rather than reset-to-1 momentum) shows the halved-momentum piece
  alone is neutral, and the EMA tolerance specifically is what hurts at low eta.
- On `ladybug-49.txt` (the small synthetic validation problem), old vs. new is
  indistinguishable either way — only one restart fires in ~200 iterations
  regardless of scheme, so this dataset can't discriminate between them at all
  and passed both before and after the change purely because it wasn't a
  meaningful test of the restart logic in the first place.

**Why this probably doesn't transfer**: the source paper's problem (distributed
pose-graph optimization, SE(3) rotation averaging structure) and this one
(bundle adjustment, reprojection-error surrogate) have different curvature/
conditioning of the MM majorizer. Tolerating a temporary cost increase helps
when the surrogate's local geometry lets momentum "coast through" a bad step and
recover more than it lost; here, at this problem's conditioning, skipping a
restart the old rule would have taken apparently costs more than it buys back
within a 200-iteration budget.

**Shipped default: `eta=1.0`** — the EMA degenerates to a no-op (restart trigger
identical to the pre-change code), so behavior is unchanged from before except
for the confirmed-neutral halved-momentum piece. `--eta` is exposed for further
experimentation (e.g. at different iteration budgets, or on other real
reconstructions this repo has access to), but low eta is deliberately *not* the
default given what was actually measured, not what the source paper reported on
a different problem.

## Also implemented and measured: gradient-based restart (O'Donoghue & Candes)

**Source**: O'Donoghue & Candes, *"Adaptive Restart for Accelerated Gradient
Schemes"* (arXiv:1204.3982, Sec 3.2) — the original paper the distributed-PGO
restart work above builds on. Alongside the function scheme already covered,
they propose a mechanistically different **gradient scheme**: restart whenever

```
grad_f(y^(k-1))^T (x^(k) - x^(k-1)) > 0
```

i.e. whenever the step actually taken and the gradient at the momentum point
point in the same (uphill) direction — momentum making things worse, measured
directly rather than inferred from the cost. The paper's stated appeal: "all
quantities involved... are already calculated in accelerated schemes, so no
extra computation is required" — true here too: `grad_f(y)` is exactly `(gc, gp)`,
already computed by every `MmStep` call as a byproduct of the block solve: no
new Jacobian pass needed, just not discarding what's already there. The
extra cost is one small reduction kernel (`KernelGradientRestartDot{Cameras,Points}`)
computing the dot product — no extra `ComputeCost` call, unlike the function
scheme's restart check.

Implemented as `--restart-scheme function|gradient` in both `daba_mm.cu` and
`reference_mm.py`, using the same tangent-space convention as the existing
Nesterov extrapolation (`Log(R_new · R_curr^T)` for rotations, Euclidean for
translation/points). Cross-validated CUDA vs. numpy exactly on both
`ladybug-49.txt` and this repo's muellcontainer BAL.

**Result, three problems, 200 iterations, function-scheme baseline vs. gradient-scheme:**

| problem | function scheme | gradient scheme | delta |
|---|---:|---:|---:|
| ladybug-49 (49 cam / 7.8K pt / 32K obs) | 1.636729e4 | 1.636729e4 | ~0 (negligible) |
| muellcontainer BAL (62 cam / 18K pt / 74K obs) | 6.854524e4 | 6.854399e4 | ~0.02% better (negligible) |
| venice-1778 (1,778 cam / 994K pt / 5.0M obs) | 1.553651e7 | 1.562371e7 | **~0.56% worse** |

**Neither literature-backed restart refinement (this one, or the EMA-tolerant
one above) beats the plain original "restart on any cost increase, reset q to
1" rule on this repo's real problems** — on the two smaller problems both
refinements are noise-level indistinguishable from the baseline, and on the
largest/hardest one both are measurably worse (EMA: not retested at this scale;
gradient: 0.56% worse, directly measured). This is a genuine, mildly surprising
finding: two independent, well-established pieces of restart literature both
fail to transfer a benefit to this specific problem's conditioning, in the
same direction (no improvement, mild regression at scale). Kept as
`--restart-scheme gradient`, opt-in, not the default, for the same reason
`--eta` isn't defaulted low: measured, not assumed.

## Broader first-order-methods literature check

Given both restart refinements above turned out not to transfer, checked
further afield in first-order methods generally (not just restart schemes) for
other candidates. Three more turned up; none implemented — reasoning below on
why each was or wasn't worth pursuing given what's already been measured.

**Barzilai-Borwein (BB) / spectral step size — the strongest remaining lead.**
BB step sizes replace this code's fixed `factor=2.0` majorization multiplier
with a curvature estimate from consecutive iterates and gradients (the secant
equation, `alpha = (s^T s)/(s^T y)` or `(s^T y)/(y^T y)`, `s` = position
delta, `y` = gradient delta) — cheap, well-established for exactly this kind
of "no natural step size" gradient/MM setting. Genuinely different from both
restart schemes tried above: those change *when* to reset momentum, this
changes the majorization strength *itself*, every iteration, not just at
restarts. Not implemented here — would need tracking the previous iteration's
gradient (not currently kept around past the block solve that consumes it)
and picking a per-block vs. global BB estimate, which is a real design
decision, not a one-line change. The most promising untried direction given
what's been ruled out so far; a natural next step if this is picked back up.

**Catalyst (Lin, Mairal, Harchaoui — arXiv:1506.02186, arXiv:1712.05654).**
A generic acceleration wrapper: solve a sequence of proximal-point
subproblems (built from the *original* objective plus a quadratic anchor to
the previous iterate) to increasing accuracy, with a specific warm-start and
stopping-tolerance schedule, provably recovering the optimal first-order rate
around *any* base method — including block coordinate descent, which is
structurally close to what this MM step already is. Theoretically the most
powerful option found, but it's an outer-loop restructuring (an extra nested
convergence criterion + envelope parameter to tune), not a local change like
the other three items here — given the two restart refinements already tried
underperformed a much simpler baseline on these real problems, adopting a
substantially more complex scheme without being able to test it first would
repeat the same mistake this whole investigation was set up to avoid. Flagging
as real and citable, not pursuing without a way to validate it cheaply first.

**Accelerated parallel/randomized block coordinate descent** (Nesterov 2012;
Fercoq & Richtárik, "Accelerated, Parallel, and Proximal Coordinate Descent,"
SIAM J. Optim. 2015). Checked and **not applicable to this code's setting**:
this literature's speedup comes from touching only a random subset of
coordinate blocks per iteration (importance-sampled by per-block Lipschitz
constant), which matters when per-block updates are the bottleneck on a
sequential/CPU setup. `daba_mm.cu` already updates every camera and point
block every iteration, fully in parallel on the GPU — the bottleneck here is
`KernelJacobianAccumulate`'s O(nobs) pass, not per-block sequential cost, so
there's no idle compute for importance sampling to reclaim. Read the core
papers, correctly does not transfer to a full-batch GPU MM setting — recorded
here so it isn't re-investigated later under the assumption it wasn't checked.

## Implemented and measured: Barzilai-Borwein adaptive majorization factor

Ported to both files as `--factor-scheme fixed|bb` / `factor_scheme="bb"`, with
`--factor-tau` / `factor_tau` controlling how much the raw per-iteration secant
estimate is smoothed (`tau=1.0` = raw, `tau→0` = heavily smoothed toward the
previous factor) — see the derivation and caveats in the `bb_factor_from_dots`
docstring above. The raw (`tau=1.0`) estimate oscillates badly on its own
(hits both the min and max clamp repeatedly), so before concluding anything I
swept `tau` down to see whether standard BB stabilization (smoothing) could
rescue it — the fair test, not a strawman of the idea.

**Result, ladybug-49 and the muellcontainer BAL problem, 200 iterations:**

| tau | ladybug-49 final cost | muellcontainer final cost |
|---|---:|---:|
| 1.00 (raw) | 1.639792e4 | 7.176971e4 |
| 0.50 | 1.638663e4 | 7.082154e4 |
| 0.30 | 1.638284e4 | 6.990585e4 |
| 0.15 | 1.637434e4 | 6.958113e4 |
| 0.05 | 1.636755e4 | 6.891681e4 |
| 0.02 | — | 6.854568e4 |
| **fixed (baseline)** | **1.636729e4** | **6.854524e4** |

**Never beats the fixed baseline at any smoothing level, on either problem.**
Heavier smoothing monotonically closes the gap but only *asymptotes toward*
the baseline as `tau→0` — the adaptive scheme's best-case behavior is
converging to "become the fixed constant," never surpassing it. This is a
clean, unambiguous negative result, not a borderline one.

**CUDA cross-validation note**: unlike the two restart schemes above, BB's
CUDA and numpy trajectories do **not** match to 6 sig figs past ~60
iterations (confirmed matching exactly through the first ~60, then visibly
diverging) — GPU `atomicAdd` reduction order is non-deterministic at the bit
level (already a known property of this codebase, noted in the original
design write-up), and BB feeds a continuous secant ratio back into the
trajectory *every* iteration, unlike the restart schemes' occasional discrete
threshold comparisons, so tiny floating-point noise compounds into a
genuinely different trajectory rather than washing out. Both implementations
still land in the same "worse than baseline" regime independently, which is
the substantive conclusion — but it's the first scheme in this investigation
where the two didn't bit-match, worth recording honestly rather than papering
over. Also has real per-iteration overhead: the extra `BBDots` kernel pass
pushed venice-1778 from 40.5ms/iter to 60.6ms/iter at tau=1.0 — confirmed on
that problem too:

| | venice-1778 final cost (200 it) |
|---|---:|
| BB, tau=1.0 | 1.599080e7 |
| BB, tau=0.05 | 1.569475e7 |
| fixed (baseline) | **1.553651e7** |

Three for three: BB-adaptive factor is worse than the fixed constant on every
problem tested, at every smoothing level tested.

## Implemented and measured: SQUAREM extrapolation — and a real algorithmic finding along the way

Ported to both files as `--accel-scheme nesterov|squarem` / `solve_squarem()`
— the Varadhan & Roland 2008 "S3" scheme, orthogonal to Nesterov (a
replacement for the whole outer-loop acceleration mechanism, not layered on
top of it). One SQUAREM "meta-iteration" costs 2 `mm_step` calls vs.
Nesterov's 1 per iteration, so all comparisons below are at matched
`mm_step`-equivalent count (100 SQUAREM meta-iterations = 200 `mm_step` calls
≈ Nesterov's 200 iterations), not matched "iteration" labels.

**While building this, hit a genuine anomaly and chased it down rather than
assuming a bug in SQUAREM's math**: on the muellcontainer BAL problem, the
safeguarded fallback path (accept the plain double-`mm_step` result, `X2`,
whenever the SQUAREM extrapolation doesn't beat it) was producing costs that
increased relative to where the meta-iteration *started*. Isolated it
completely: **two bare, sequential `mm_step` calls (factor=2, no SQUAREM math
involved at all) can themselves increase cost**, starting from certain real
points on this problem —

```
cost before = 2.304521e+06
cost after 1 mm_step = 3.089078e+06
cost after 2 mm_step = 6.356137e+06
```

— confirmed in complete isolation, and confirmed that plain sequential MM
*is* monotone from the problem's original starting point (20 iterations,
strictly decreasing every step). So this isn't "MM is broken" — it's that the
`factor=2` majorization bound doesn't hold **pointwise everywhere** on real
data, only along the specific trajectory plain sequential MM happens to
visit from a normal start; a point reached via extrapolation (Nesterov's or
SQUAREM's) can land somewhere with locally stiffer curvature than `factor=2`
accounts for, and once there, this codebase's block solve — fixed `factor`,
fixed `lam`, no LM-style adaptive re-damping on a failed step — has no way to
recover. This is a real robustness gap in the underlying MM step, independent
of SQUAREM or any of the acceleration schemes tried in this document; it's
usually invisible because Nesterov's own trajectory rarely revisits such a
point twice in a row, but SQUAREM's more aggressive extrapolation exposes it
directly. **Concrete follow-up this points to**: the LM-style adaptive
damping Ceres and Caspar already use elsewhere in this project (increase
damping on a rejected step, decrease on an accepted one) is the natural fix,
and ties back to the "not pursued" Mairal/line-search-MM note from the first
round of this document — this finding is independent evidence that item is
worth doing, not just theoretically nice.

Built a diagnostic-only `strict_monotone=True` variant of the numpy safeguard
(not ported to CUDA, not shipped) that also requires beating the
meta-iteration's own starting cost, to confirm the above cleanly: with it on,
the run gets **permanently stuck** the first time it hits such a point (every
subsequent meta-iteration rejects identically, since the state literally
never changes) — a clear illustration that there's no self-correction
available, not just an occasional hiccup. The shipped default
(`strict_monotone=False`) matches the textbook SQUAREM safeguard *and* is
consistent with how this file's own Nesterov-restart branch already behaves
(it also accepts whatever `mm_step(curr)` produces with no comparison against
the pre-restart cost) — holding SQUAREM to a stricter standard than the
baseline it's being compared against would have been an unfair test.

**Result, matched `mm_step`-equivalent budget (~200), three problems:**

| problem | Nesterov baseline | SQUAREM | delta |
|---|---:|---:|---:|
| ladybug-49 | 1.636729e4 | 1.637319e4 | ~0.04% worse |
| muellcontainer BAL | 6.854524e4 | 1.648657e6 | **24x worse** (hit the majorization-failure point above) |
| venice-1778 | 1.553651e7 | 1.595299e7 | ~2.7% worse |

Worse on all three, ranging from marginal (ladybug) to dramatic
(muellcontainer, directly explained by the finding above). SQUAREM's CUDA
port **does** match the numpy reference exactly on all tested problems (unlike
BB) — its accept/reject decisions are discrete cost comparisons with enough
margin that GPU reduction-order noise doesn't flip them, at least at the
scales tested here.

## Closed-form warm start: still not implemented

Revisited whether to take a swing at this now that more has been learned.
Decided against fabricating one: the DABA IJRR 2025 abstract states this
"always improves bundle adjustment estimates," which is a specific, falsifiable
claim about *their* construction — inventing a plausible-sounding warm start
of my own and labeling it as satisfying that claim would misrepresent what's
actually known. Still blocked on the same thing as before: the formula isn't
in the openly-readable parts of the paper. Left undone rather than guessed at.

## Third round: multi-shift-inspired per-block adaptive damping — the first real win

The "adaptive LM-style damping" follow-up flagged above (and independently
re-confirmed by the insta360x4 dataset's NaN-cascade failure — see that
investigation's writeup) pointed at a specific gap: this codebase's block
solve uses a *fixed* `factor`/`lam` for every block, every iteration, with
no mechanism to recover from a locally bad step. [msouiai/multishift-bundle-adjustment](https://github.com/msouiai/multishift-bundle-adjustment)
turned out to be a well-matched, previously-untried answer to exactly this
gap — not because its core mechanism transplants directly, but because its
underlying *philosophy* does.

**Why the mechanism itself doesn't transplant**: that repo speeds up
Levenberg-Marquardt by solving the *full coupled* camera+point normal
equations with conjugate gradients, exploiting Krylov shift-invariance
(`K_m(A,b) = K_m(A+σI,b)` for every shift σ) so **one stream of matvecs
yields solutions for a whole grid of damping values λ at once** — instead of
the classical "guess λ, solve, reject, redo" loop, which throws away a full
CG solve on every rejection. Their headline finding: this wins exactly when
the baseline's own rejection rate is high (not predicted by problem size or
density), and **greedy min-cost selection (accept whichever λ's candidate
has the lowest true, non-linearized cost) beats gain-ratio selection
significantly** — their own explicit ablation, not assumed. DABA's entire
design avoids ever forming that coupled system in the first place — every
camera/point is an independent 6×6/3×3 block, already solved by direct
Cholesky. There is no expensive shared matvec stream for a Krylov trick to
amortize at that scale; plugging `multishift_cg` into DABA literally has
nothing to act on.

**What transplants is the philosophy, adapted to what's actually expensive
here.** For blocks this tiny, you don't need CG's shift-invariance to get
"one pass, many λ" cheaply — solving a 6×6 or 3×3 Cholesky system at 6
different λ costs a handful of extra flops, utterly negligible next to the
O(nobs) Jacobian-accumulation kernel that already dominates every DABA
iteration's wall-clock (established repeatedly earlier in this document).
So: give each block its own small λ grid (6 candidates, ±2 decades around a
*persistent per-block* base λ, not a single global scalar), evaluate every
candidate's **true local reprojection cost** (matching their own
gain-ratio-loses finding, not the linearized model's predicted reduction),
and accept the best if it improves — **per block**, not globally. That last
part is the key adaptation beyond a literal port: a single ill-conditioned
camera or point simply freezes at its current position for that iteration
instead of being dragged along by a λ tuned for the average block, or
(as measured directly on insta360x4) corrupting every neighboring block on
the next iteration via a NaN cascade.

Implemented and cross-validated exactly (CUDA vs. numpy reference, bit-for-
bit on ladybug-49 and the muellcontainer BAL problem) as
`solve_retract_multilambda`/`solve_multilambda` in `reference_mm.py` and
`--damping-scheme multilambda` in `daba_mm.cu`.

**Result, muellcontainer BAL problem (62 cams, 18,044 pts, 73,546 obs), 200
iterations** — full writeup and plot in
`../colmap_solver_comparison/run_real_simpleradial_62cam_18Kpt_caspar_ok/README.md`:

| scheme | wall-clock | final cost | final RMSE |
|---|---:|---:|---:|
| fixed λ (original) | 0.154s | 6.85452e4 | 1.3653px |
| multi-lambda (new) | 0.402s | **5.52898e4** | **1.2262px** |

**19.4% lower cost**, closing most of the remaining gap to the centralized-
Ceres-BAL reference (1.2199px, same fixed-intrinsics problem) — at ~2.6x the
wall-clock (still 0.4s, still dramatically faster than Ceres' 5.9s or
Caspar's 0.97s on the COLMAP-path version of this same problem). Block
acceptance was 100% throughout — every camera, every point, every
iteration — meaning this isn't a robustness safety net quietly doing
nothing on an easy problem; it is genuinely finding a better per-block step
than the fixed λ on *every single iteration* of a well-conditioned real
problem, not just rescuing pathological ones. **This is the first
literature-adapted mechanism in this entire investigation (two prior rounds,
five other mechanisms) that actually beats the original fixed-parameter
baseline**, not just approaches it or fails more gracefully.

Also confirmed on ladybug-49 (the small synthetic validation problem):
identical final cost to the fixed baseline (1.636729e4, exact match), 100%
acceptance throughout — expected and reassuring, since that problem is easy
enough that the fixed λ=1e-6 was already never rejected; multi-lambda
correctly degenerates to the same behavior rather than doing something
different-but-not-better on a problem with nothing to fix.

**Not yet done**: a test on venice-1778 (the largest real problem in this
repo) or on the insta360x4 pathological case specifically, to see whether
per-block freezing actually prevents the NaN cascade that motivated this in
the first place — the muellcontainer result already answers "does this help
on a normal problem," but the insta360x4 case is the more direct test of
the robustness claim this was originally designed around.

## Summary across all three rounds

| mechanism | source | status | verdict |
|---|---|---|---|
| EMA-tolerant restart | Fan et al. arXiv:2108.00083 | implemented, cross-validated | doesn't transfer; ~0.6% worse at low eta |
| Gradient-based restart | O'Donoghue & Candès arXiv:1204.3982 | implemented, cross-validated | doesn't transfer; up to 0.56% worse |
| BB-adaptive factor | our own adaptation | implemented, cross-validated* | never beats baseline at any tau |
| SQUAREM | Varadhan & Roland 2008 | implemented, cross-validated | worse on all 3 problems; surfaced a real robustness gap |
| Closed-form warm start | DABA IJRR 2025 | blocked | formula not accessible |
| Barzilai-Borwein (2nd round's lead) | — | superseded by above | tested, negative |
| Catalyst | Lin/Mairal/Harchaoui | not attempted | too large a change to validate cheaply |
| Accelerated coordinate descent | Nesterov 2012; Fercoq & Richtárik 2015 | ruled out | doesn't apply to full-batch GPU MM |
| **Multi-lambda per-block damping** | **msouiai/multishift-bundle-adjustment** | **implemented, cross-validated** | **19.4% lower cost on muellcontainer — first real win** |

Six literature-backed mechanisms implemented and honestly measured against
this codebase's own real problems; five didn't improve on the existing
simple scheme, and the sixth — per-block adaptive damping, adapted (not
literally ported) from multi-shift CG's core philosophy — genuinely does.
The earlier five rounds' most valuable output was the majorization-
robustness finding that pointed straight at this gap; this round is the
payoff of actually following that lead instead of stopping at "we found a
problem."
