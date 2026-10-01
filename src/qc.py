"""Per-cell quality metrics and filtering."""

from __future__ import annotations

import anndata as ad
import pandas as pd
import scanpy as sc

from .config import Config


def annotate_qc(adata: ad.AnnData, cfg: Config) -> None:
    mito = str(cfg.req("qc.mito_prefix"))
    ribo = tuple(cfg.req("qc.ribo_prefixes"))
    adata.var["mt"] = adata.var_names.str.startswith(mito)
    adata.var["ribo"] = adata.var_names.str.startswith(ribo)
    if adata.var["mt"].sum() == 0:
        print(f"[warn] no genes start with {mito!r}; check qc.mito_prefix "
              "(human 'MT-', mouse 'mt-') and whether var_names are symbols not Ensembl IDs")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo"], percent_top=None, log1p=False, inplace=True
    )


def qc_table(adata: ad.AnnData, sample_key: str, stage: str) -> pd.DataFrame:
    """Per-sample QC summary, so cell loss at each stage is auditable."""
    g = adata.obs.groupby(sample_key, observed=True)
    out = pd.DataFrame({
        "n_cells": g.size(),
        "median_genes": g["n_genes_by_counts"].median(),
        "median_counts": g["total_counts"].median(),
        "median_pct_mt": g["pct_counts_mt"].median(),
        "median_pct_ribo": g["pct_counts_ribo"].median(),
    })
    out.insert(0, "stage", stage)
    return out.reset_index()


def filter_cells(adata: ad.AnnData, cfg: Config) -> tuple[ad.AnnData, pd.DataFrame]:
    """Apply QC thresholds; return the filtered object and a per-sample audit of losses.

    The n_genes upper bound (a crude doublet proxy) is a per-sample quantile rather than
    one global value — see the note in configs/scrna.yaml.
    """
    sample_key = cfg.key("sample")
    min_genes = int(cfg.req("min_genes"))
    max_mito_pct = float(cfg.req("max_mito_fraction")) * 100.0
    quant = float(cfg.req("qc.max_genes_quantile"))

    caps = adata.obs.groupby(sample_key, observed=True)["n_genes_by_counts"].quantile(quant)
    cap_per_cell = adata.obs[sample_key].map(caps).values

    fail_genes = adata.obs["n_genes_by_counts"].values < min_genes
    fail_mito = adata.obs["pct_counts_mt"].values > max_mito_pct
    fail_cap = adata.obs["n_genes_by_counts"].values > cap_per_cell
    keep = ~(fail_genes | fail_mito | fail_cap)

    audit = pd.DataFrame({
        sample_key: adata.obs[sample_key].values,
        "fail_min_genes": fail_genes,
        "fail_max_mito": fail_mito,
        "fail_upper_cap": fail_cap,
        "kept": keep,
    }).groupby(sample_key, observed=True).sum(numeric_only=True).reset_index()
    audit["n_cells_in"] = (
        adata.obs.groupby(sample_key, observed=True).size().reindex(audit[sample_key]).values
    )

    out = adata[keep].copy()
    n_genes_before = out.n_vars
    sc.pp.filter_genes(out, min_cells=int(cfg.req("qc.min_cells_per_gene")))
    print(f"cells {adata.n_obs} -> {out.n_obs}; genes {n_genes_before} -> {out.n_vars}")

    floor = int(cfg.req("qc.min_cells_per_sample"))
    per_sample = out.obs[sample_key].value_counts()
    too_small = per_sample[per_sample < floor].index.tolist()
    if too_small:
        print(f"[warn] samples below qc.min_cells_per_sample={floor} and dropped: {too_small}")
        out = out[~out.obs[sample_key].isin(too_small)].copy()
        out.obs[sample_key] = out.obs[sample_key].astype(str).astype("category")
    return out, audit
