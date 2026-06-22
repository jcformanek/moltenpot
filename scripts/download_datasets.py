"""
scripts/download_datasets.py — Pre-download Moltenpot datasets from HuggingFace Hub.

Reads scenario lists directly from configs/train_offline.yaml so there is
no duplication between config and script. Run this before training to fetch
all required datasets in one go, rather than being prompted per-scenario
during training.

Usage
-----
# Download all in-dist + OOD scenarios for clean_up (non-interactive)
python scripts/download_datasets.py \\
    --data-root data/clean_up \\
    --substrate clean_up \\
    --yes

# Download only in-distribution scenarios
python scripts/download_datasets.py \\
    --data-root data/clean_up \\
    --substrate clean_up \\
    --split in_dist \\
    --yes

# Download specific scenarios by name
python scripts/download_datasets.py \\
    --data-root data/clean_up \\
    --scenarios clean_up_0 clean_up_1 clean_up_6 \\
    --yes

# Use a custom HF repo
python scripts/download_datasets.py \\
    --data-root data/clean_up \\
    --substrate clean_up \\
    --repo-id yourname/moltenpot-datasets \\
    --yes
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "train_offline.yaml",
)


def _load_scenario_lists(substrate: str) -> tuple[list[str], list[str]]:
    """Read in_dist and out_dist scenario lists from train_offline.yaml."""
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(_CONFIG_PATH)
        in_dist  = list(cfg.in_dist_scenarios)
        out_dist = list(cfg.out_dist_scenarios) if cfg.out_dist_eval else []
        return in_dist, out_dist
    except Exception as exc:
        print(f"[warn] Could not read scenario lists from config: {exc}")
        return [], []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download Moltenpot datasets from HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Local directory to download into, e.g. data/clean_up",
    )
    parser.add_argument(
        "--substrate", default=None,
        help="Substrate name — used to look up default scenario lists from config",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Explicit scenario names to download (overrides --substrate lookup)",
    )
    parser.add_argument(
        "--split", choices=["all", "in_dist", "out_dist"], default="all",
        help="Which split to download when using --substrate (default: all)",
    )
    parser.add_argument(
        "--repo-id", default=None,
        help="HuggingFace dataset repo ID (default: value in hub.py / train_offline.yaml)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Download without interactive prompts (for cluster / CI use)",
    )
    args = parser.parse_args()

    from moltenpot import hub

    # Resolve repo_id: CLI flag > config file > hub.py default
    repo_id = args.repo_id
    if repo_id is None:
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(_CONFIG_PATH)
            repo_id = cfg.hub.repo_id
        except Exception:
            pass  # fall through to hub.py default
    hub.configure(repo_id=repo_id, auto_download=args.yes)

    # Resolve scenario list
    if args.scenarios:
        scenarios = args.scenarios
    elif args.substrate:
        in_dist, out_dist = _load_scenario_lists(args.substrate)
        if not in_dist and not out_dist:
            print(
                f"No scenarios found in config for substrate '{args.substrate}'. "
                "Pass --scenarios explicitly."
            )
            sys.exit(1)
        if args.split == "in_dist":
            scenarios = in_dist
        elif args.split == "out_dist":
            scenarios = out_dist
        else:
            scenarios = in_dist + out_dist
    else:
        parser.error("Provide either --substrate or --scenarios.")

    print(f"Checking {len(scenarios)} scenario(s) in {os.path.abspath(args.data_root)} ...")
    hub.ensure_datasets(args.data_root, scenarios)
    print("All datasets available.")


if __name__ == "__main__":
    main()
