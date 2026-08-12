# MM convergence literature review: what could improve `daba_mm.cu`

Checked the majorization-minimization (MM) and first-order-methods acceleration
literature against what `daba_mm.cu` currently does, to see if there are concrete,
non-speculative improvements available rather than just tuning existing knobs.
Two rounds: MM-specific literature first (three papers with direct findings,
ranked by strength of evidence + ease of implementation below), then broader
first-order methods generally (further down). Two candidates from the first
round were actually implemented, cross-validated, and measured — both turned
out not to help on this repo's real problems, contrary to their source papers'
own findings on different problem classes; see those sections for the numbers.

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

## Next steps not yet tried

- **Barzilai-Borwein majorization factor** (above) — the strongest untried
  lead from this round; changes the per-iteration step strength rather than
  the restart logic, genuinely orthogonal to everything tried so far.
- Item 2 (closed-form warm start) and item 3 (SQUAREM), from the first round,
  remain unimplemented — neither depends on any restart-scheme outcome.
- Whether low `eta` or the gradient scheme help at a much larger iteration
  budget (1000+, vs. the 200 used everywhere in this repo for comparability)
  wasn't tested. Given the venice-1778 result at 200 iterations went the wrong
  direction for the gradient scheme specifically, this is a lower-priority
  follow-up than BB, not a first thing to retry.
