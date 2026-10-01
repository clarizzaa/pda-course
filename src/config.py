"""Config loading and run bookkeeping.

All parameters come from configs/*.yaml (WORKSPACE.md rule 1). Nothing in src/ or
workflows/ may define a numeric default: if a key is missing from the YAML, that is
an error, not something to paper over.
"""

from __future__ import annotations

import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """dict with dotted lookup that raises on a missing key instead of returning None."""

    def req(self, dotted: str) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"configs: required key {dotted!r} is missing")
            node = node[part]
        if node is None:
            raise KeyError(f"configs: required key {dotted!r} is null — set it before running")
        return node

    def opt(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def path(self, dotted: str) -> Path:
        """Resolve a config path relative to the repo root."""
        return (REPO_ROOT / str(self.req(dotted))).resolve()

    def key(self, name: str) -> str:
        """An obs column name from the `keys:` block."""
        return self.req(f"keys.{name}")


def load_config(name: str = "scrna") -> Config:
    path = REPO_ROOT / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        return Config(yaml.safe_load(fh))


def set_seed(cfg: Config) -> int:
    """Seed every RNG the pipeline touches. Scanpy calls also take random_state explicitly."""
    seed = int(cfg.req("random_seed"))
    random.seed(seed)
    np.random.seed(seed)
    return seed


def outdirs(cfg: Config) -> dict[str, Path]:
    dirs = {name: cfg.path(f"paths.{name}") for name in ("processed", "results", "figures")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_runlog(cfg: Config, step: str, extra: dict | None = None) -> Path:
    """Record the config and commit a step actually ran under, next to its outputs."""
    log = {
        "step": step,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "config": dict(cfg),
        **(extra or {}),
    }
    out = cfg.path("paths.results") / f"runlog_{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2, default=str))
    return out
