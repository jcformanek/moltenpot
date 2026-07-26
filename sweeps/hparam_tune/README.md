# Setting-3 leak re-tune — hyper-parameter sweeps

## Why

Hyper-parameters were originally tuned on **scenario 0** of each substrate and kept
fixed across all settings. Two Setting-3 splits put that tuning scenario in their
**test** set, which is a test-set leak:

| Split | Substrate | Tuning scenario in test set | Re-tune on (smallest train index) |
|---|---|---|---|
| **C4** | coins | `coins_0` | **`coins_5`** |
| **A1** | allelopathic_harvest | `allelopathic_harvest__open_0` | **`allelopathic_harvest__open_1`** |

(Commons Harvest was tuned on `commons_harvest__closed_0`, which is never in a
test set — no leak. Clean Up and Coop Mining are also clean.)

For each affected split we re-tune on the **smallest-index scenario in that split's
train set**, mirroring the original single-scenario tuning protocol. The winning
hyper-parameters will later be used for a clean re-run of C4 / A1 in Setting 3
(done separately).

## What each run does

- Single-scenario training on the tuning scenario only, **10,000 updates**.
- **No eval during training** (`eval_at_end=true` skips the step-0 and interval evals).
- **One final eval of 100 episodes** on the same training scenario (`num_eval_workers=4`;
  a single scenario means one Ray task actually runs).
- Logs `eval/in_dist/return_mean` at the end — the metric to rank on.

Scenario config lives in `configs/experiment/tune_coins_5.yaml` and
`configs/experiment/tune_allelopathic_open_1.yaml`. Both scenarios run inside each
sweep via the swept `+experiment` parameter, and are separated in W&B by group
(`tune_coins_5` / `tune_allelopathic_open_1`).

## Sweeps (one per algorithm, grid search)

| Algo | Swept params | Runs |
|---|---|--:|
| BC  | scenario ×2, LR ×3 | 6 |
| CQL | scenario ×2, LR ×3, `cql_alpha` ∈ {2,5,10} | 18 |
| BCQ | scenario ×2, LR ×3, `tau` ∈ {0.1,0.3,0.5} (q_lr = imitator_lr) | 18 |
| IQL | scenario ×2, LR ×3, `beta` ∈ {1,3,5} (v_lr = q_lr = actor_lr) | 18 |
| | **Total** | **60** |

LR grid for every algo: `{3e-4, 1e-4, 5e-5}`.

## Launch

From the repo root, with the env active and the W&B entity set:

```bash
source .venv_311/bin/activate
export WANDB_ENTITY=moltenpot          # runs land in moltenpot/moltenpot-offline-all

# 1. Register each sweep — each prints a <sweep_id>
wandb sweep sweeps/hparam_tune/bc.yaml
wandb sweep sweeps/hparam_tune/cql.yaml
wandb sweep sweeps/hparam_tune/bcq.yaml
wandb sweep sweeps/hparam_tune/iql.yaml

# 2. Launch one or more agents per sweep (point many machines at the same ID)
wandb agent moltenpot/moltenpot-offline-all/<bc_sweep_id>
wandb agent moltenpot/moltenpot-offline-all/<cql_sweep_id>
wandb agent moltenpot/moltenpot-offline-all/<bcq_sweep_id>
wandb agent moltenpot/moltenpot-offline-all/<iql_sweep_id>
```

Datasets auto-download on first use (coins data is not present locally; allelopathic
open_1 is). Grid sweeps run every combination, so an agent exits once its sweep is
exhausted — add more agents to parallelise across compute.

## Picking the winners

For each `(algorithm, scenario)` pair, choose the hyper-parameter combination with the
highest final `eval/in_dist/return_mean`. Group by the W&B group (`tune_coins_5` /
`tune_allelopathic_open_1`) and read the end-of-run metric. Those winning params then
feed the clean C4 / A1 Setting-3 re-runs (separate step).

## Notes

- **Shared LR** is wired via OmegaConf interpolation in the sweep command
  (`algorithm.imitator_lr=${algorithm.q_lr}`, etc.), resolved at runtime — so a single
  swept LR drives all of an algorithm's learning rates. Verified with `--cfg job --resolve`.
- No code paths change when these flags are off; `eval_at_end` and the tuning configs are
  additive.
