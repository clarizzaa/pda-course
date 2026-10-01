"""Reading data/raw, attaching sample metadata, and validating the experimental design."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

from .config import Config


def load_matrix(path: Path) -> ad.AnnData:
    """Read .h5ad, 10x .h5, or a directory holding an MTX triplet. Raw counts expected."""
    if path.is_dir():
        adata = sc.read_10x_mtx(path, var_names="gene_symbols", cache=False)
    elif path.suffix == ".h5ad":
        adata = sc.read_h5ad(path)
    elif path.suffix == ".h5":
        adata = sc.read_10x_h5(path)
    else:
        raise ValueError(
            f"unsupported input {path.name!r}: expected .h5ad, 10x .h5, or an MTX directory"
        )
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


def assert_raw_counts(adata: ad.AnnData) -> None:
    """Refuse pre-normalised input: pseudobulk DE is invalid on logCPM values."""
    sub = adata.X[: min(1000, adata.n_obs)]
    vals = sub.data if hasattr(sub, "data") else sub.ravel()
    if vals.size == 0:
        raise ValueError("matrix has no non-zero entries")
    frac_int = float((vals == vals.astype(int)).mean())
    if frac_int < 0.99:
        raise ValueError(
            f"X does not look like raw counts (only {frac_int:.1%} of sampled values are "
            "integers). Point paths.matrix at the raw count matrix — normalisation happens "
            "in workflow 02."
        )


def attach_sample_meta(adata: ad.AnnData, meta_path: Path | None, sample_key: str) -> None:
    """Left-join a sample-level table onto obs by sample key."""
    if meta_path is None:
        return
    meta = pd.read_csv(meta_path)
    if sample_key not in meta.columns:
        raise KeyError(f"{sample_key!r} not a column of {meta_path.name}: {list(meta.columns)}")
    if meta[sample_key].duplicated().any():
        raise ValueError(f"{meta_path.name} has duplicate {sample_key} rows; expected one per sample")
    meta = meta.set_index(sample_key)
    unmatched = set(adata.obs[sample_key].unique()) - set(meta.index)
    if unmatched:
        raise ValueError(f"samples in the matrix with no metadata row: {sorted(unmatched)}")
    for col in meta.columns:
        adata.obs[col] = adata.obs[sample_key].map(meta[col]).values


def check_design(adata: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Validate labels and return the sample-level design table.

    Fails loudly on missing labels. A silent NaN here would propagate into every group
    contrast downstream and be invisible in the results.
    """
    sample_key, response_key = cfg.key("sample"), cfg.key("response")
    for key in (sample_key, response_key):
        if key not in adata.obs:
            raise KeyError(f"obs has no column {key!r}; available: {list(adata.obs.columns)}")
        n_missing = int(adata.obs[key].isna().sum())
        if n_missing:
            raise ValueError(f"{n_missing} cells carry no {key!r} label")

    groups = sorted(adata.obs[response_key].astype(str).unique())
    if len(groups) != 2:
        raise ValueError(f"{response_key!r} must have exactly 2 levels, found {groups}")

    design = (
        adata.obs.groupby([sample_key, response_key], observed=True)
        .size().rename("n_cells").reset_index()
    )
    for extra in ("patient", "timepoint", "batch"):
        col = cfg.opt(f"keys.{extra}")
        if col and col in adata.obs:
            first = adata.obs.groupby(sample_key, observed=True)[col].first()
            design[extra] = design[sample_key].map(first)

    n_per_group = design.groupby(response_key, observed=True)[sample_key].nunique()
    if (n_per_group < int(cfg.req("de.min_samples_per_group"))).any():
        print(
            "[warn] fewer samples per response group than de.min_samples_per_group:\n"
            f"{n_per_group.to_string()}\n"
            "       group comparisons are descriptive; say so in the report"
        )
    return design


def check_confounding(design: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cross-tabulate response against batch (WORKSPACE.md rule 7).

    A batch that is perfectly nested in a response group makes the two effects
    inseparable — no amount of correction recovers them, and the report has to say so.
    """
    batch_key = cfg.opt("keys.batch")
    if not batch_key or batch_key not in design.columns:
        return pd.DataFrame()
    tab = pd.crosstab(design[batch_key], design[cfg.key("response")])
    nested = (tab > 0).sum(axis=1).eq(1).all()
    tab.attrs["fully_nested"] = bool(nested)
    if nested:
        print(f"[warn] every {batch_key} level sits in exactly one response group: "
              "batch and response are confounded and cannot be separated")
    return tab
