"""Figures. Every function takes an explicit output path and returns it."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scanpy as sc             # noqa: E402
import seaborn as sns           # noqa: E402

from .config import Config      # noqa: E402

DPI = 200


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def qc_violins(adata: ad.AnnData, cfg: Config, out: Path) -> Path:
    """Per-sample QC distributions with the configured thresholds drawn on."""
    sample_key = cfg.key("sample")
    metrics = [("n_genes_by_counts", float(cfg.req("min_genes"))),
               ("total_counts", None),
               ("pct_counts_mt", float(cfg.req("max_mito_fraction")) * 100)]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(
        max(6, 0.45 * adata.obs[sample_key].nunique() + 3), 3.0 * len(metrics)), sharex=True)
    for ax, (metric, thresh) in zip(np.atleast_1d(axes), metrics):
        sns.violinplot(data=adata.obs, x=sample_key, y=metric, ax=ax, cut=0,
                       inner="quartile", linewidth=0.6)
        if metric == "total_counts":
            ax.set_yscale("log")
        if thresh is not None:
            ax.axhline(thresh, color="crimson", ls="--", lw=1)
        ax.set_xlabel("")
    for lab in np.atleast_1d(axes)[-1].get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
    fig.suptitle("QC per sample (dashed = configured threshold)", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def umap_panels(adata: ad.AnnData, cfg: Config, cluster_key: str, out: Path) -> Path:
    keys = [cluster_key, "cell_type", cfg.key("sample"), cfg.key("response")]
    keys = [k for k in keys if k in adata.obs]
    ncol = 2
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 5.5 * nrow))
    for ax, key in zip(np.atleast_1d(axes).ravel(), keys):
        sc.pl.umap(adata, color=key, ax=ax, show=False, frameon=False,
                   legend_fontsize=7, title=key, size=6)
    for ax in np.atleast_1d(axes).ravel()[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out)


def composition_box(long: pd.DataFrame, cfg: Config, out: Path,
                    celltype_key: str = "cell_type") -> Path:
    """Per-sample fractions by response group; each point is one sample, not one cell."""
    response_key = cfg.key("response")
    order = (long.groupby(celltype_key, observed=True)["fraction"]
             .median().sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(max(7, 1.05 * len(order) + 3), 4.6))
    sns.boxplot(data=long, x=celltype_key, y="fraction", hue=response_key, order=order,
                ax=ax, showfliers=False, width=0.62, linewidth=0.8)
    sns.stripplot(data=long, x=celltype_key, y="fraction", hue=response_key, order=order,
                  ax=ax, dodge=True, size=3.6, color="0.15", legend=False)
    ax.set_xlabel("")
    ax.set_ylabel("fraction of sample")
    ax.set_title("Cell-type composition by response group (one point = one sample)",
                 fontsize=11)
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
    fig.tight_layout()
    return _save(fig, out)


def signature_violins(adata: ad.AnnData, cfg: Config, score_cols: list[str],
                      compartment: str, out: Path,
                      celltype_key: str = "cell_type") -> Path | None:
    """Signature scores in one compartment: per-sample means over a per-cell violin.

    The violin shows the cell-level spread; the overlaid points are the per-sample means,
    which is the level the test in src/stats.py operates on.
    """
    sample_key, response_key = cfg.key("sample"), cfg.key("response")
    sub = adata.obs[adata.obs[celltype_key].astype(str) == compartment]
    if sub.empty:
        print(f"[warn] no {compartment!r} cells; signature figure skipped")
        return None
    per_sample = (sub.groupby([sample_key, response_key], observed=True)[score_cols]
                  .mean().reset_index())
    fig, axes = plt.subplots(1, len(score_cols), figsize=(3.2 * len(score_cols), 3.9))
    for ax, col in zip(np.atleast_1d(axes), score_cols):
        sns.violinplot(data=sub, x=response_key, y=col, ax=ax, cut=0,
                       inner=None, linewidth=0.7, color="0.85")
        sns.stripplot(data=per_sample, x=response_key, y=col, ax=ax,
                      size=6, color="crimson", edgecolor="black", linewidth=0.5)
        ax.set_title(col.replace("sig_", ""), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("score")
    fig.suptitle(f"{compartment}: signature scores by response "
                 "(grey = cells, red = sample means)", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def de_volcano(res: pd.DataFrame, cell_type: str, cfg: Config, out: Path,
               contrast_label: str = "", n_label: int = 10) -> Path:
    """Volcano for one pseudobulk contrast. `contrast_label` fixes the axis direction."""
    alpha = float(cfg.req("de.alpha"))
    df = res.dropna(subset=["padj", "log2FoldChange"]).copy()
    df["sig"] = df["padj"] < alpha
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.scatter(df.loc[~df["sig"], "log2FoldChange"], -np.log10(df.loc[~df["sig"], "padj"]),
               s=7, color="0.75", linewidths=0)
    ax.scatter(df.loc[df["sig"], "log2FoldChange"], -np.log10(df.loc[df["sig"], "padj"]),
               s=11, color="crimson", linewidths=0)
    ax.axhline(-np.log10(alpha), color="0.4", ls="--", lw=0.8)
    top = df[df["sig"]].reindex(
        df[df["sig"]]["log2FoldChange"].abs().sort_values(ascending=False).index
    ).head(n_label)
    for gene, row in top.iterrows():
        ax.annotate(gene, (row["log2FoldChange"], -np.log10(row["padj"])),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    direction = contrast_label.replace("_vs_", " vs ") if contrast_label else "contrast"
    ax.set_xlabel(f"log2 fold change ({direction})")
    ax.set_ylabel(r"$-\log_{10}$ adjusted $p$")
    ax.set_title(f"{cell_type} — pseudobulk DE, n={int(df['sig'].sum())} at FDR<{alpha}",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, out)
