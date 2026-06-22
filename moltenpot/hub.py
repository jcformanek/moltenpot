"""
hub.py — HuggingFace Hub integration for Moltenpot datasets.
=============================================================

Handles downloading datasets from the HuggingFace Hub and uploading local
datasets to the Hub for sharing.

Dataset layout on HuggingFace (nested — substrate/scenario):
    <substrate>/<scenario>/dataset.hdf5   e.g.  clean_up/clean_up_0/dataset.hdf5

The substrate name is derived from the last component of data_root:
    data_root = "data/clean_up"  →  substrate = "clean_up"

Local layout:
    <data_root>/<scenario>/dataset.hdf5   e.g.  data/clean_up/clean_up_0/dataset.hdf5

Configure once at startup
-------------------------
    from moltenpot import hub
    hub.configure(repo_id="jcformanek/moltenpot", auto_download=True)

Auto-download on first use
--------------------------
    from moltenpot.hub import ensure_datasets
    ensure_datasets("data/clean_up", ["clean_up_0", "clean_up_1"])
    # → prompts once if any are missing, then downloads sequentially
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Update this before publishing the dataset.
HF_REPO_ID = "jcformanek/moltenpot"

# Module-level config — set once at startup via configure().
_repo_id: str = HF_REPO_ID
_auto_download: bool = True


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def configure(
    repo_id: Optional[str] = None,
    auto_download: Optional[bool] = None,
) -> None:
    """
    Set module-level defaults used by ensure_dataset / ensure_datasets.

    Call this once at the top of train_offline.py (or any entry point) so
    all downstream data-loading code picks up the right repo and behaviour.

    Parameters
    ----------
    repo_id : str, optional
        HuggingFace dataset repo ID, e.g. ``"yourname/moltenpot-datasets"``.
    auto_download : bool, optional
        If True, missing datasets are downloaded without prompting the user.
        Useful for cluster / CI runs.
    """
    global _repo_id, _auto_download
    if repo_id is not None:
        _repo_id = repo_id
    if auto_download is not None:
        _auto_download = auto_download


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _substrate(data_root: str) -> str:
    """Derive substrate name from data_root (its last path component)."""
    return os.path.basename(os.path.abspath(data_root))


def local_hdf5_path(data_root: str, scenario: str) -> str:
    """Return the expected local path: ``<data_root>/<scenario>/dataset.hdf5``."""
    return os.path.join(data_root, scenario, "dataset.hdf5")


def repo_hdf5_path(data_root: str, scenario: str) -> str:
    """Return the HuggingFace repo path: ``<substrate>/<scenario>/dataset.hdf5``."""
    return f"{_substrate(data_root)}/{scenario}/dataset.hdf5"


def is_available(data_root: str, scenario: str) -> bool:
    """Return True if the scenario HDF5 file exists on disk."""
    return os.path.isfile(local_hdf5_path(data_root, scenario))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset(
    data_root: str,
    scenario: str,
    repo_id: Optional[str] = None,
) -> str:
    """
    Download a single scenario's HDF5 file from HuggingFace Hub.

    The file is written directly to ``<data_root>/<scenario>/dataset.hdf5``
    — the HuggingFace cache directory is bypassed so datasets always live
    in the project's own ``data/`` tree.

    Parameters
    ----------
    data_root : str
        Root directory for this substrate, e.g. ``"data/clean_up"``.
    scenario : str
        Scenario key, e.g. ``"clean_up_0"``.
    repo_id : str, optional
        HF dataset repo to download from (defaults to module-level _repo_id).

    Returns
    -------
    str
        Absolute path to the downloaded file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download datasets. "
            "Install it with:  pip install huggingface_hub"
        )

    repo = repo_id or _repo_id
    dest = local_hdf5_path(data_root, scenario)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    hf_path = repo_hdf5_path(data_root, scenario)
    logger.info("Downloading %s from %s ...", hf_path, repo)

    # hf_hub_download writes the file preserving its repo path relative to
    # local_dir. With the flat layout substrate/scenario/dataset.hdf5,
    # downloading into the parent of data_root resolves to
    # data_root/scenario/dataset.hdf5 — matching `dest`.
    parent = os.path.dirname(os.path.abspath(data_root))
    hf_hub_download(
        repo_id=repo,
        filename=hf_path,
        repo_type="dataset",
        local_dir=parent,
    )

    logger.info("Saved → %s", dest)
    return dest


def ensure_dataset(
    data_root: str,
    scenario: str,
    auto_download: Optional[bool] = None,
    repo_id: Optional[str] = None,
) -> str:
    """
    Return the local path for a scenario dataset, downloading from HF if needed.

    Issues an interactive Y/n prompt when the file is missing (unless
    ``auto_download`` is True or the module was configured with
    ``hub.configure(auto_download=True)``).

    Raises
    ------
    FileNotFoundError
        If the dataset is missing and the user declines to download.
    """
    if is_available(data_root, scenario):
        return local_hdf5_path(data_root, scenario)

    _auto = auto_download if auto_download is not None else _auto_download
    repo  = repo_id or _repo_id
    path  = local_hdf5_path(data_root, scenario)

    print(f"\n[moltenpot] Dataset not found locally: {path}")

    if not _auto:
        try:
            answer = input(f"  Download '{scenario}' from {repo}? [Y/n]: ")
        except EOFError:
            answer = "n"
        if answer.strip().lower() in ("n", "no"):
            raise FileNotFoundError(
                f"Dataset '{scenario}' not found at {path}. "
                "Run `python scripts/download_datasets.py` to fetch it."
            )

    return download_dataset(data_root, scenario, repo_id=repo)


def ensure_datasets(
    data_root: str,
    scenarios: List[str],
    auto_download: Optional[bool] = None,
    repo_id: Optional[str] = None,
) -> None:
    """
    Check all scenarios and offer to download any that are missing in one batch.

    Issues a single prompt listing all missing scenarios (rather than one
    prompt per file), then downloads them sequentially with progress output.

    Parameters
    ----------
    data_root : str
        Root directory for this substrate.
    scenarios : list of str
        All scenario keys that should be present.
    auto_download : bool, optional
        Override the module-level auto_download setting for this call.
    repo_id : str, optional
        Override the module-level repo_id for this call.

    Raises
    ------
    FileNotFoundError
        If any datasets are missing and the user declines to download.
    """
    missing = [s for s in scenarios if not is_available(data_root, s)]
    if not missing:
        return

    _auto = auto_download if auto_download is not None else _auto_download
    repo  = repo_id or _repo_id

    print(f"\n[moltenpot] {len(missing)} dataset(s) not found locally:")
    for s in missing:
        print(f"    {local_hdf5_path(data_root, s)}")
    print(f"  Source: {repo} (HuggingFace Hub)")

    if not _auto:
        try:
            answer = input(f"\n  Download {len(missing)} dataset(s)? [Y/n]: ")
        except EOFError:
            answer = "n"
        if answer.strip().lower() in ("n", "no"):
            raise FileNotFoundError(
                f"Missing datasets: {missing}. "
                "Run `python scripts/download_datasets.py` to fetch them."
            )

    for i, scenario in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {scenario}/dataset.hdf5 ...", end=" ", flush=True)
        download_dataset(data_root, scenario, repo_id=repo)
        print("done")

    print(f"  All {len(missing)} dataset(s) downloaded to {os.path.abspath(data_root)}/\n")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_dataset(
    data_root: str,
    scenario: str,
    repo_id: Optional[str] = None,
    *,
    create_repo: bool = False,
    private: bool = True,
) -> None:
    """
    Upload a single scenario's HDF5 file to HuggingFace Hub.

    The local file ``<data_root>/<scenario>/dataset.hdf5`` is uploaded to
    ``<substrate>/<scenario>/dataset.hdf5`` in the Hub repo.

    Parameters
    ----------
    data_root : str
        Root directory containing the scenario subdirectory.
    scenario : str
        Scenario key to upload.
    repo_id : str, optional
        Target HF dataset repo (defaults to module-level _repo_id).
    create_repo : bool
        If True, create the repo on HF if it does not exist.
    private : bool
        Whether a newly created repo should be private (default True).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to upload datasets. "
            "Install it with:  pip install huggingface_hub"
        )

    repo  = repo_id or _repo_id
    local = local_hdf5_path(data_root, scenario)

    if not os.path.isfile(local):
        raise FileNotFoundError(f"Dataset file not found: {local}")

    api = HfApi()

    if create_repo:
        api.create_repo(
            repo_id=repo,
            repo_type="dataset",
            exist_ok=True,
            private=private,
        )
        logger.info("Repo ready: https://huggingface.co/datasets/%s", repo)

    hf_path = repo_hdf5_path(data_root, scenario)
    size_gb = os.path.getsize(local) / (1024 ** 3)
    logger.info("Uploading %s (%.1f GB) → %s/%s ...", scenario, size_gb, repo, hf_path)

    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=hf_path,
        repo_id=repo,
        repo_type="dataset",
    )
    logger.info("Uploaded %s", hf_path)
