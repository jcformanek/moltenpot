# Setting 2 · Clean Up — normalised in-dist return (mean ± SEM over 9 scenarios × seeds)

Normalisation per scenario: (return − random) / (p90 − random), baselines from outputs/tier2_returns.json.

| Condition | BC | BCQ | IQL | CQL | **Aggregate (all algos)** |
|---|---|---|---|---|---|
| seq_len = 1 | 0.381 ± 0.095 | 0.358 ± 0.099 | 0.532 ± 0.159 | 0.084 ± 0.063 | **0.321 ± 0.052** |
| seq_len = 4 | 0.640 ± 0.054 | 1.063 ± 0.168 | 0.666 ± 0.138 | 0.237 ± 0.084 | **0.614 ± 0.061** |
| seq_len = 128 (standard) | 0.554 ± 0.045 | 0.959 ± 0.115 | 0.879 ± 0.128 | -0.046 ± 0.042 | **0.586 ± 0.059** |
| scenario-ID (seq_len 128) | — | — | 0.841 ± 0.146 | — | IQL only → 0.841 ± 0.146 |

Seed counts (n): seq_len = 1: 3/3/2/3 (bc/bcq/iql/cql), seq_len = 4: 3/2/3/3 (bc/bcq/iql/cql), seq_len = 128: 3/3/3/3 (bc/bcq/iql/cql)

1 = dataset p90 (best-in-data), 0 = random policy. Values can exceed 1 or go below 0.
