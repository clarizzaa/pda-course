"""Normalisation, feature selection, embedding, clustering."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .config import Config


def normalise(adata: ad.AnnData, cfg: Config) -> None:
    """CP10K + log1p, keeping raw counts in layers['counts'].

    The counts layer is not a convenience: pseudobulk DE in workflow 05 must sum counts,
    and summing normalised values would discard the library-size information DESeq2
    needs to model dispersion.
    """
    if cfg.req("normalization") != "log1p":
        raise NotImplementedError(f"normalization={cfg.req('normalization')!r} not implemented")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=float(cfg.req("norm.target_sum")))
    sc.pp.log1p(adata)
    adata.raw = adata          # log-normalised values for marker tests and scoring


def select_hvgs(adata: ad.AnnData, cfg: Config) -> int:
    batch_key = cfg.key(cfg.req("norm.hvg_batch_key_from"))
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=int(cfg.req("norm.n_hvg")),
        flavor=str(cfg.req("norm.hvg_flavor")),
        layer="counts",
        batch_key=batch_key,
    )
    pattern = str(cfg.req("norm.exclude_hvg_pattern"))
    drop = adata.var_names.str.match(pattern)
    n_dropped = int((adata.var["highly_variable"] & drop).sum())
    adata.var.loc[drop, "highly_variable"] = False
    print(f"HVGs: {int(adata.var['highly_variable'].sum())} "
          f"({n_dropped} excluded by norm.exclude_hvg_pattern)")
    return int(adata.var["highly_variable"].sum())


def embed(adata: ad.AnnData, cfg: Config, seed: int) -> str:
    """Scale, PCA, and optionally Harmony. Returns the obsm key to cluster on.

    Integration touches the embedding only. Patient identity is confounded with response
    in this design, so correcting the expression matrix itself would partly erase the
    effect the analysis is testing for — all differential testing uses uncorrected values.
    """
    sc.pp.scale(adata, max_value=float(cfg.req("norm.scale_max_value")), zero_center=True)
    sc.tl.pca(adata, n_comps=int(cfg.req("n_pcs")), svd_solver="arpack",
              mask_var="highly_variable", random_state=seed)

    mode = str(cfg.req("embed.integrate"))
    if mode == "none":
        return "X_pca"
    if mode != "harmony":
        raise ValueError(f"embed.integrate must be 'harmony' or 'none', got {mode!r}")
    key = cfg.key(cfg.req("embed.integrate_key_from"))
    run_harmony(adata, key, int(cfg.req("n_pcs")), seed)
    return "X_pca_harmony"


def run_harmony(adata: ad.AnnData, key: str, n_pcs: int, seed: int,
                basis: str = "X_pca", out_key: str = "X_pca_harmony") -> None:
    """Harmony on `basis`, written to obsm[out_key].

    harmonypy is called directly rather than through sc.external.pp.harmony_integrate:
    harmonypy <2 returned Z_corr as (dims, cells) and the scanpy wrapper transposes it,
    but harmonypy >=2 already returns (cells, dims), so the wrapper hands anndata a
    transposed array and the assignment fails. Checking the orientation here works with
    either version. Verified against harmonypy 2.0.2 / scanpy 1.12.4.
    """
    import harmonypy

    X = np.asarray(adata.obsm[basis], dtype=np.float64)
    out = harmonypy.run_harmony(X, adata.obs, key, random_state=seed, verbose=False)
    Z = np.asarray(out.Z_corr)
    if Z.shape == (adata.n_obs, n_pcs):
        corrected = Z
    elif Z.shape == (n_pcs, adata.n_obs):
        corrected = Z.T
    else:
        raise RuntimeError(
            f"harmonypy returned Z_corr with shape {Z.shape}; expected "
            f"({adata.n_obs}, {n_pcs}) or its transpose"
        )
    adata.obsm[out_key] = np.ascontiguousarray(corrected)


def cluster(adata: ad.AnnData, cfg: Config, rep: str, seed: int) -> str:
    """Neighbours, UMAP, and a Leiden resolution sweep. Returns the primary cluster key."""
    sc.pp.neighbors(adata, n_neighbors=int(cfg.req("embed.n_neighbors")),
                    use_rep=rep, random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    for res in cfg.req("embed.leiden_resolutions"):
        sc.tl.leiden(adata, resolution=float(res), key_added=f"leiden_{res}",
                     flavor="igraph", n_iterations=2, directed=False, random_state=seed)
    primary = f"leiden_{cfg.req('embed.leiden_primary')}"
    if primary not in adata.obs:
        raise KeyError(f"embed.leiden_primary={primary} is not among the resolutions swept")
    return primary


def resolution_sweep_table(adata: ad.AnnData, cfg: Config) -> pd.DataFrame:
    """Cluster counts per resolution, so the chosen resolution is defensible."""
    rows = []
    for res in cfg.req("embed.leiden_resolutions"):
        key = f"leiden_{res}"
        sizes = adata.obs[key].value_counts()
        rows.append({
            "resolution": res,
            "n_clusters": int(sizes.size),
            "min_cluster_size": int(sizes.min()),
            "median_cluster_size": float(sizes.median()),
            "primary": key == f"leiden_{cfg.req('embed.leiden_primary')}",
        })
    return pd.DataFrame(rows)
