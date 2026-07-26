# Launch runbook — Setting-3 leak re-tune sweeps

A step-by-step guide for launching the four hyper-parameter tuning sweeps across
compute. Self-contained: you should not need any other file to execute this.

## What these sweeps are

Two Setting-3 splits tuned hyper-parameters on a scenario that appears in their own
test set (a leak): **C4** (coins, leaked `coins_0`) and **A1** (allelopathic,
leaked `allelopathic_harvest__open_0`). We re-tune on the smallest-index scenario in
each split's *train* set — **`coins_5`** and **`allelopathic_harvest__open_1`** —
then (later, separately) re-run C4 / A1 with the winning params.

Each run: single-scenario train, 10,000 updates, **no eval during training**, then
**one 100-episode eval** on the training scenario. Rank on the final
`eval/in_dist/return_mean`.

## The four sweeps (grid search, one per algorithm)

| File | Algo | Swept params | Runs |
|---|---|---|--:|
| `sweeps/hparam_tune/bc.yaml`  | BC  | scenario ×2, LR ×3 | 6 |
| `sweeps/hparam_tune/cql.yaml` | CQL | scenario ×2, LR ×3, `cql_alpha`∈{2,5,10} | 18 |
| `sweeps/hparam_tune/bcq.yaml` | BCQ | scenario ×2, LR ×3, `tau`∈{0.1,0.3,0.5} | 18 |
| `sweeps/hparam_tune/iql.yaml` | IQL | scenario ×2, LR ×3, `beta`∈{1,3,5} | 18 |

LR grid (all algos): `{3e-4, 1e-4, 5e-5}`. **Total across all four sweeps: 60 runs.**
Both scenarios run inside each sweep via the swept `+experiment` param, separated in
W&B by group: `tune_coins_5` and `tune_allelopathic_open_1`.

## Prerequisites (do these first)

1. **Repo up to date.** The checkout must include this work: `sweeps/hparam_tune/`,
   `configs/experiment/tune_coins_5.yaml`, `configs/experiment/tune_allelopathic_open_1.yaml`,
   and the `eval_at_end` flag in `configs/train_offline.yaml` + the four
   `moltenpot/algorithms/*.py`. On branch `develop`. If these files are missing,
   `git pull` first.
2. **Python env active** (project venv), e.g.:
   ```bash
   source .venv_311/bin/activate
   ```
3. **W&B auth + entity.** Log in (`wandb login`) if not already, and set the entity so
   runs land in `moltenpot/moltenpot-offline-all`:
   ```bash
   export WANDB_ENTITY=moltenpot
   ```
4. **Data.** Datasets auto-download on first use (coins data is not local by default;
   allelopathic `open_1` may already be present). No manual download step needed.

Run everything **from the repo root**.

## Step 1 — Register each sweep and capture its ID

`wandb sweep <file>` prints a sweep ID and the exact `wandb agent ...` line to use.
Register all four:

```bash
wandb sweep sweeps/hparam_tune/bc.yaml
wandb sweep sweeps/hparam_tune/cql.yaml
wandb sweep sweeps/hparam_tune/bcq.yaml
wandb sweep sweeps/hparam_tune/iql.yaml
```

Each prints something like:
```
wandb: Created sweep with ID: ab12cd34
wandb: Run sweep agent with: wandb agent moltenpot/moltenpot-offline-all/ab12cd34
```
**Record the four sweep IDs** (or the full `moltenpot/moltenpot-offline-all/<id>` paths).
To capture programmatically, the ID is on stderr; e.g.:
```bash
wandb sweep sweeps/hparam_tune/bc.yaml 2>&1 | tee /tmp/bc_sweep.txt
BC_SWEEP=$(grep -oE 'moltenpot/moltenpot-offline-all/[a-z0-9]+' /tmp/bc_sweep.txt | tail -1)
```

## Step 2 — Launch agents

Point one or more agents at each sweep ID. Agents on **different machines** using the
**same** ID pull work from the same grid — that is how you parallelise.

```bash
wandb agent moltenpot/moltenpot-offline-all/<bc_sweep_id>
wandb agent moltenpot/moltenpot-offline-all/<cql_sweep_id>
wandb agent moltenpot/moltenpot-offline-all/<bcq_sweep_id>
wandb agent moltenpot/moltenpot-offline-all/<iql_sweep_id>
```

Guidance:
- A grid sweep is finite; each agent **exits automatically** once the grid is exhausted.
- Add more agents (same ID) to go faster. Each run uses one GPU (if present) plus up to
  4 CPUs for the final eval, so size agents-per-machine to your hardware.
- BC has only 6 runs; CQL/BCQ/IQL have 18 each.

## Step 3 — Completion and picking winners

- A sweep is done when all its runs show `finished` in W&B and its agents have exited.
- For each `(algorithm, scenario)` pair, pick the hyper-parameter combination with the
  **highest final `eval/in_dist/return_mean`**. Filter by W&B group
  (`tune_coins_5` vs `tune_allelopathic_open_1`) and read the end-of-run value.
- You will end up with 8 winning configs: {BC, CQL, BCQ, IQL} × {coins_5, allelo_open_1}.
  These feed the clean C4 / A1 Setting-3 re-runs (a separate step — not part of this).

## Troubleshooting

- **`Could not override 'experiment'`** or the run ignores the scenario: the `+experiment`
  param must reach Hydra as `+experiment=<name>`. It is defined with the `+` in the sweep
  YAML; don't strip it.
- **BCQ/IQL learning rates look wrong** (e.g. imitator/v/actor LR not matching q_lr): the
  sweeps tie them via `algorithm.imitator_lr=${algorithm.q_lr}` (and v_lr/actor_lr) in the
  command. These resolve at runtime via Hydra/OmegaConf. If your W&B version mangles the
  `${...}`, replace those command lines with the literal LR by making the extra LR keys
  swept params equal to `algorithm.q_lr`'s values.
- **Dataset download prompts**: runs auto-download via the HF hub; ensure the machine has
  network access and HF availability. No credentials needed for the public datasets.
- **Nothing logs at the end**: confirm `eval_at_end=true` is set (it is, in the two tune
  configs) and `num_eval_workers=4` (>0) so the final eval actually runs.
