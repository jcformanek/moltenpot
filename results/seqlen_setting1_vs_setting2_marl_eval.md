# Clean Up — Setting 1 vs Setting 2 seq_len ablation (marl-eval aggregate)

Aggregate scores from the **marl-eval** pipeline (rliable: median / IQM / mean /
optimality-gap, 50k stratified-bootstrap 95% CIs), computed on per-scenario
**normalised** final-eval returns, J~_s = (R_s − random_s)/(p90_s − random_s),
baselines from `outputs/tier2_returns.json`. Pooled over the 9 in-dist clean_up
scenarios (`benchmark_clean_ALL.in_dist_scenarios`). Same protocol as the paper's
Table 2, restricted to clean_up, with the Setting-2 seq_len ablation added.

## View B — aggregated across all algorithms (each condition pooled over BC/BCQ/IQL/CQL × seeds)

| Condition | Median | IQM | Mean | Optimality Gap ↓ |
|---|---|---|---|---|
| Setting 1 (per-scenario) | 0.79 [0.65, 0.97] | 0.70 [0.59, 0.84] | 0.77 [0.66, 0.89] | 0.42 [0.36, 0.48] |
| Setting 2 · seq_len 128 | 0.58 [0.44, 0.78] | 0.51 [0.39, 0.63] | 0.59 [0.49, 0.68] | 0.52 [0.46, 0.59] |
| Setting 2 · seq_len 4 | 0.57 [0.44, 0.69] | 0.58 [0.51, 0.65] | 0.61 [0.54, 0.69] | 0.49 [0.44, 0.54] |
| Setting 2 · seq_len 1 | 0.35 [0.23, 0.46] | 0.28 [0.23, 0.34] | 0.32 [0.27, 0.37] | 0.70 [0.67, 0.74] |

Higher is better for Median/IQM/Mean; lower is better for Optimality Gap. 1 = dataset p90, 0 = random policy.

## View A — per algorithm × condition

### Mean [95% CI]

| Algo | Setting 1 (per-scenario) | Setting 2 · seq_len 128 | Setting 2 · seq_len 4 | Setting 2 · seq_len 1 |
|---|---|---|---|---|
| BC | 0.61 [0.60, 0.63] | 0.55 [0.53, 0.58] | 0.64 [0.62, 0.66] | 0.38 [0.36, 0.41] |
| BCQ | 1.37 [1.29, 1.46] | 0.96 [0.84, 1.07] | 1.06 [1.01, 1.12] | 0.36 [0.28, 0.44] |
| IQL | 0.93 [0.90, 0.95] | 0.88 [0.83, 0.93] | 0.67 [0.65, 0.68] | 0.53 [0.50, 0.57] |
| CQL | 0.18 [0.15, 0.22] | -0.05 [-0.07, -0.02] | 0.24 [0.20, 0.27] | 0.08 [0.06, 0.10] |

### OptGap ↓ [95% CI]

| Algo | Setting 1 (per-scenario) | Setting 2 · seq_len 128 | Setting 2 · seq_len 4 | Setting 2 · seq_len 1 |
|---|---|---|---|---|
| BC | 0.39 [0.37, 0.40] | 0.45 [0.42, 0.47] | 0.36 [0.35, 0.38] | 0.62 [0.60, 0.65] |
| BCQ | 0.16 [0.14, 0.17] | 0.26 [0.20, 0.33] | 0.27 [0.22, 0.32] | 0.67 [0.61, 0.74] |
| IQL | 0.32 [0.29, 0.34] | 0.34 [0.29, 0.39] | 0.50 [0.49, 0.51] | 0.56 [0.54, 0.58] |
| CQL | 0.82 [0.78, 0.85] | 1.05 [1.02, 1.07] | 0.76 [0.73, 0.80] | 0.92 [0.90, 0.94] |

Seed counts (n runs pooled per condition): Setting 1 (per-scenario) = 12, Setting 2 · seq_len 128 = 12, Setting 2 · seq_len 4 = 11, Setting 2 · seq_len 1 = 11.
Missing seeds: IQL seq_len=1 (n=2), BCQ seq_len=4 (n=2); all others n=3 per algo.
