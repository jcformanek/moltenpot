"""
moltenpot.algorithms — Offline RL algorithm registry.
"""

from moltenpot.algorithms.bc  import train as train_bc
from moltenpot.algorithms.bcq import train as train_bcq
from moltenpot.algorithms.iql import train as train_iql
from moltenpot.algorithms.cql import train as train_cql
from moltenpot.algorithms.icq import train as train_icq

ALGORITHMS = {
    "bc":  train_bc,
    "bcq": train_bcq,
    "iql": train_iql,
    "cql": train_cql,
    "icq": train_icq,
}

__all__ = ["ALGORITHMS", "train_bc", "train_bcq", "train_iql", "train_cql", "train_icq"]
