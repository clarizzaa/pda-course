"""Marker-based cluster annotation and signature scoring."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .config import Config


def score_panels(adata: ad.AnnData, panels: dict[str, list[str]], prefix: str,
                 min_genes_present: int = 2) -> list[str]:
    """sc.tl.score_genes for each panel, skipping panels too sparsely represented.

    Returns the obs columns created. A panel reduced to one detected gene is not a
    signature score, so it is skipped loudly rather than scored.
    """
    cols: list[str] = []
    for name, genes in panels.items():
        present = [g for g in genes if g in adata.raw.var_names]
        if len(present) < min_genes_present:
            print(f"[warn] panel {name!r}: {len(present)}/{len(genes)} genes detected, skipped")
            continue
        if len(present) < len(genes):
            missing = sorted(set(genes) - set(present))
            print(f"[note] panel {name!r} scored on {len(present)}/{len(genes)} genes "
                  f"(absent: {', '.join(missing)})")
        sc.tl.score_genes(adata, present, score_name=f"{prefix}{name}", use_raw=True)
        cols.append(f"{prefix}{name}")
    if not cols:
        raise ValueError(f"no panel with prefix {prefix!r} could be scored; "
                         "are var_names gene symbols?")
    return cols


def label_clusters(adata: ad.AnnData, cluster_key: str, score_cols: list[str],
                   prefix: str, out_key: str = "cell_type") -> pd.DataFrame:
    """Label each cluster by its top-scoring panel; return the full score matrix.

    The argmax runs across panels *within* a cluster, on the raw score_genes values.
    It deliberately does not run on scores z-scored across clusters: score_genes already
    subtracts a matched control gene set, so raw panel scores are comparable within a
    cluster, whereas z-scoring across clusters normalises away exactly the magnitude
    difference that separates overlapping panels. CD8_T (CD3D/CD3E/CD8A/CD8B) and CD4_T
    (CD3D/CD3E/CD4/IL7R) share half their genes, so a CD8 cluster elevates both panels;
    on raw scores CD8_T wins on 4/4 genes against 2/4, but after z-scoring both panels
    look equally "high in one cluster" and the pick is close to arbitrary.

    The per-cluster z-scores are still returned as `z_*` columns — they answer the other
    question ("which cluster is most enriched for this panel") and are useful for review.

    The argmax is a first pass, not an answer. Read `margin_to_runner_up`: a cluster
    whose top two panels nearly tie needs manual adjudication, the call recorded in
    reports/DECISIONS.md and set in configs/manual_labels.yaml.
    """
    raw = adata.obs.groupby(cluster_key, observed=True)[score_cols].mean()
    ordered = np.sort(raw.values, axis=1)
    winner = raw.idxmax(axis=1).str.slice(len(prefix))
    z = (raw - raw.mean()) / raw.std(ddof=0).replace(0, np.nan)

    out = raw.copy()
    out.columns = [c[len(prefix):] for c in raw.columns]
    for col in z.columns:
        out[f"z_{col[len(prefix):]}"] = z[col]
    out["assigned"] = winner
    out["top_score"] = ordered[:, -1]
    out["margin_to_runner_up"] = ordered[:, -1] - ordered[:, -2]
    out["n_cells"] = adata.obs[cluster_key].value_counts().reindex(raw.index).values

    adata.obs[out_key] = adata.obs[cluster_key].map(winner).astype("category")
    return out.reset_index()


def cluster_markers(adata: ad.AnnData, cluster_key: str, n_genes: int = 50) -> pd.DataFrame:
    """Wilcoxon one-vs-rest per cluster — for *identifying* clusters, not for group tests.

    This test treats cells as independent, which is fine for naming a cluster and wrong
    for comparing response groups (WORKSPACE.md rule 6); that contrast is workflow 05.
    """
    sc.tl.rank_genes_groups(adata, cluster_key, method="wilcoxon", use_raw=True)
    return sc.get.rank_genes_groups_df(adata, group=None).groupby(
        "group", observed=True, group_keys=False
    ).head(n_genes)


def apply_manual_labels(adata: ad.AnnData, mapping: dict[str, str],
                        cluster_key: str, out_key: str = "cell_type") -> None:
    """Override automatic labels from a cluster -> label dict (e.g. loaded from YAML)."""
    unknown = set(mapping) - set(adata.obs[cluster_key].astype(str).unique())
    if unknown:
        raise KeyError(f"manual labels reference clusters not present: {sorted(unknown)}")
    current = adata.obs[out_key].astype(str)
    clusters = adata.obs[cluster_key].astype(str)
    for cl, label in mapping.items():
        current[clusters == cl] = label
    adata.obs[out_key] = current.astype("category")
