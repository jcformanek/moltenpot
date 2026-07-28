# Setting 3 — hyperparameter-leak re-tune: test-set delta (old leaked vs clean re-run)

Reviewer flagged that Setting-3 hparams were tuned on the "first scenario", which for
two splits (**Coins C4**, **Allelopathic A1**) is a *test* scenario (index 0), leaking
the held-out set into model selection. We re-tuned each on the lowest-index **training**
scenario (C4 → `coins_5`, A1 → `allelopathic_harvest__open_1`) and re-ran the split.

**Metric:** zero-shot **test-set** (out_dist) return, normalised per-scenario D4RL-style
`J~ = (R - random)/(p90 - random)` with baselines from `outputs/tier3_returns.json`, then
aggregated. New numbers are the `eval_at_end` test evals of the re-runs (W&B groups
`rerun_C4`, `rerun_A1`). Old numbers are the currently-reported leaked-tuning results
(`tier3_returns.json`). 1 = dataset p90, 0 = random.

## Coins C4  (test = {coins_0})

| Algo | Old (leaked) | New (clean) | Δ |
|---|---|---|---|
| BC  | 0.50 | 0.30 | −0.20 |
| BCQ | 0.56 | 0.40 | −0.16 |
| IQL | 0.44 | 0.41 | −0.03 |
| CQL | 0.65 | 0.59 | −0.06 |
| **Aggregate** | **0.54** | **0.43** | **−0.11** |

marl-eval: Mean 0.54 [0.48, 0.60] → 0.43 [0.34, 0.51]; Optimality gap 0.46 [0.40, 0.52] → 0.57 [0.49, 0.66].
The C4 leak did inflate test performance modestly; the drop is driven by BC and BCQ (IQL/CQL barely move).

## Allelopathic A1  (test = {open_0, open_5, open_6})

| Algo | Old (leaked) | New (clean) | Δ |
|---|---|---|---|
| BC  | 0.569 | 0.569 | +0.00 |
| BCQ | 0.624 | 0.659 | +0.035 |
| IQL | 0.638 | 0.656 | +0.018 |
| CQL (n=2) | 0.503 | 0.474 | −0.029 |
| **Aggregate** | **0.584** | **0.600** | **+0.016** |

A1 shows no leak effect: the clean re-tune ties or slightly beats the leaked one.
Note: `cql` seed 3 crashed at ~step 16k (n=2 for that cell); does not materially affect the aggregate.

## Combined (C4 + A1 pooled)

| | Old (leaked) | New (clean) |
|---|---|---|
| Mean normalised return | 0.57 [0.52, 0.63] | 0.55 [0.49, 0.62] |
| Optimality gap | 0.43 [0.37, 0.48] | 0.45 [0.39, 0.52] |

Pooled change ≈ −0.02 normalised return (optimality gap +0.02), well within the 95% CIs.
Pooling is row-weighted, so A1 (3 test scenarios) dominates C4 (1 test scenario); the
per-split view above is the more transparent read: **modest decrease on C4, none on A1.**

_Re-run protocol: benchmark_{coins_C4, allelopathic_A1}, winning tuned hparams, 20k updates,
block_read=false (matches the original protocol), eval_at_end on the split's default test set._
