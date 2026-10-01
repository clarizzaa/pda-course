"""Group comparisons. The experimental unit is the sample, never the cell.

WORKSPACE.md rule 6 is the governing constraint in this module. A per-cell test over
4000 cells from one patient reports 4000 independent observations where there is really
one; every function here aggregates to the sample first and tests across samples.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

from .config import Config


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #

def celltype_fractions(adata: ad.AnnData, cfg: Config,
                       celltype_key: str = "cell_type") -> pd.DataFrame:
    """Long-form per-sample cell-type fractions, carrying each sample's total cell count.

    The total is reported alongside because a fraction from 40 cells and one from 4000
    are not comparable, and the sample n is what a reader needs to weigh the figure.
    """
    sample_key, response_key = cfg.key("sample"), cfg.key("response")
    counts = pd.crosstab(adata.obs[sample_key], adata.obs[celltype_key])
    frac = counts.div(counts.sum(axis=1), axis=0)
    long = frac.stack().rename("fraction").reset_index()
    long.columns = [sample_key, celltype_key, "fraction"]
    long["n_cells_celltype"] = counts.stack().values
    long["n_cells_sample"] = long[sample_key].map(counts.sum(axis=1)).values
    meta = adata.obs.groupby(sample_key, observed=True).first()
    for extra in ("response", "patient", "timepoint", "batch"):
        col = cfg.opt(f"keys.{extra}")
        if col and col in meta.columns:
            long[col] = long[sample_key].map(meta[col]).values
    long[response_key] = long[sample_key].map(meta[response_key]).values
    return long


def composition_response_test(long: pd.DataFrame, cfg: Config,
                              celltype_key: str = "cell_type") -> pd.DataFrame:
    """Mann-Whitney per cell type on per-sample fractions, BH-corrected.

    Screening only: fractions sum to 1, so cell types are not independent and a real
    expansion of one compartment deflates all the others. A Dirichlet-multinomial model
    (scCODA) is the confirmatory test if the assignment calls for one.
    """
    response_key = cfg.key("response")
    groups = sorted(long[response_key].astype(str).unique())
    rows = []
    for ct, sub in long.groupby(celltype_key, observed=True):
        a = sub.loc[sub[response_key].astype(str) == groups[0], "fraction"].values
        b = sub.loc[sub[response_key].astype(str) == groups[1], "fraction"].values
        if len(a) < 2 or len(b) < 2:
            rows.append({celltype_key: ct, "U": np.nan, "p": np.nan,
                         f"n_{groups[0]}": len(a), f"n_{groups[1]}": len(b),
                         "status": "too_few_samples"})
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            celltype_key: ct, "U": stat, "p": p,
            f"median_{groups[0]}": float(np.median(a)),
            f"median_{groups[1]}": float(np.median(b)),
            f"n_{groups[0]}": len(a), f"n_{groups[1]}": len(b),
            "status": "tested",
        })
    res = pd.DataFrame(rows)
    tested = res["p"].notna()
    res.loc[tested, "p_adj"] = multipletests(
        res.loc[tested, "p"], method=str(cfg.req("composition.multiple_testing"))
    )[1]
    return res.sort_values("p_adj", na_position="last").reset_index(drop=True)


def composition_paired_test(long: pd.DataFrame, cfg: Config,
                            celltype_key: str = "cell_type") -> pd.DataFrame:
    """Paired Wilcoxon on pre vs on-treatment fractions within patient.

    Returns an empty frame when the design has no usable pairs — the caller should
    report that as 'not assessed', not as a null result.
    """
    patient_key, timepoint_key = cfg.opt("keys.patient"), cfg.opt("keys.timepoint")
    if not patient_key or not timepoint_key:
        return pd.DataFrame()
    if patient_key not in long.columns or timepoint_key not in long.columns:
        return pd.DataFrame()
    tps = sorted(long[timepoint_key].dropna().astype(str).unique())
    if len(tps) != 2:
        return pd.DataFrame()

    wide = long.pivot_table(index=[patient_key, celltype_key],
                            columns=timepoint_key, values="fraction")
    rows = []
    for ct, sub in wide.groupby(celltype_key, observed=True):
        pair = sub[tps].dropna()
        if len(pair) < 3:
            continue
        stat, p = wilcoxon(pair[tps[0]].values, pair[tps[1]].values)
        rows.append({celltype_key: ct, "n_patients": len(pair), "W": stat, "p": p,
                     f"median_{tps[0]}": float(pair[tps[0]].median()),
                     f"median_{tps[1]}": float(pair[tps[1]].median())})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_adj"] = multipletests(
            res["p"], method=str(cfg.req("composition.multiple_testing"))
        )[1]
        res = res.sort_values("p_adj").reset_index(drop=True)
    return res


# --------------------------------------------------------------------------- #
# Signature scores, aggregated to the sample
# --------------------------------------------------------------------------- #

def signature_response_test(adata: ad.AnnData, cfg: Config, score_cols: list[str],
                            compartment: str, celltype_key: str = "cell_type") -> pd.DataFrame:
    """Per-sample mean signature score within one compartment, tested across samples."""
    sample_key, response_key = cfg.key("sample"), cfg.key("response")
    sub = adata.obs[adata.obs[celltype_key].astype(str) == compartment]
    if sub.empty:
        return pd.DataFrame()
    per_sample = sub.groupby(sample_key, observed=True)[score_cols].mean()
    n_cells = sub.groupby(sample_key, observed=True).size()
    resp = sub.groupby(sample_key, observed=True)[response_key].first()
    groups = sorted(resp.astype(str).unique())
    rows = []
    for col in score_cols:
        a = per_sample.loc[resp.astype(str) == groups[0], col].dropna().values
        b = per_sample.loc[resp.astype(str) == groups[1], col].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"signature": col, "compartment": compartment, "U": stat, "p": p,
                     f"mean_{groups[0]}": float(np.mean(a)),
                     f"mean_{groups[1]}": float(np.mean(b)),
                     f"n_{groups[0]}": len(a), f"n_{groups[1]}": len(b),
                     "min_cells_per_sample": int(n_cells.min())})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_adj"] = multipletests(res["p"], method="fdr_bh")[1]
    return res


# --------------------------------------------------------------------------- #
# Pseudobulk differential expression
# --------------------------------------------------------------------------- #

def make_pseudobulk(adata: ad.AnnData, cfg: Config,
                    celltype_key: str = "cell_type") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum raw counts within (sample x cell type). Returns (counts, library metadata)."""
    if "counts" not in adata.layers:
        raise KeyError("layers['counts'] missing — run workflow 02 so raw counts are kept")
    sample_key = cfg.key("sample")
    min_cells = int(cfg.req("de.min_cells_per_library"))
    X = adata.layers["counts"]
    obs = adata.obs

    libs, mats, meta = [], [], []
    groups = obs.groupby([sample_key, celltype_key], observed=True).indices
    for (samp, ct), idx in groups.items():
        if len(idx) < min_cells:
            continue
        sub = X[idx]
        tot = np.asarray(sub.sum(axis=0)).ravel() if issparse(sub) else np.asarray(sub).sum(axis=0)
        lib = f"{samp}||{ct}"
        libs.append(lib)
        mats.append(tot)
        meta.append({"library": lib, "sample": str(samp), "cell_type": str(ct),
                     "n_cells": int(len(idx))})
    if not libs:
        raise ValueError(f"no sample x cell type group reached de.min_cells_per_library={min_cells}")
    pb = pd.DataFrame(np.vstack(mats), index=libs, columns=adata.var_names)
    pb = pb.round().astype(np.int64)
    return pb, pd.DataFrame(meta).set_index("library")


def deseq2_by_celltype(pb: pd.DataFrame, pb_meta: pd.DataFrame, sample_meta: pd.DataFrame,
                       cfg: Config, outdir: Path) -> pd.DataFrame:
    """Responder vs non-responder per cell type with PyDESeq2. Writes one table per type.

    Cell types without `de.min_samples_per_group` libraries in both groups are recorded
    as skipped rather than tested; an under-powered fit reported as a null result is the
    more misleading of the two outcomes.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    # de.n_cpus > 1 makes pydeseq2 fan out over joblib/loky, which needs POSIX
    # semaphores; in a sandboxed or container environment that raises
    # PermissionError: [Errno 1] on SC_SEM_NSEMS_MAX. Serial is the safe default and
    # these pseudobulk matrices are small.
    inference = DefaultInference(n_cpus=int(cfg.req("de.n_cpus")))
    response_key = cfg.key("response")
    alpha = float(cfg.req("de.alpha"))
    min_samples = int(cfg.req("de.min_samples_per_group"))
    meta = pb_meta.join(sample_meta[[response_key]], on="sample")
    meta[response_key] = meta[response_key].astype(str)

    # The contrast is stated explicitly rather than left to pydeseq2's level ordering:
    # it fixes which direction a positive log2FoldChange means, and a sign error here
    # would invert every biological conclusion in the report.
    levels = sorted(meta[response_key].unique())
    ref = cfg.opt("de.reference_level")
    if ref is None:
        ref = levels[0]
        print(f"[note] de.reference_level not set; using {ref!r} as the baseline. "
              "Set it in configs/scrna.yaml to make the direction explicit.")
    if str(ref) not in levels:
        raise ValueError(f"de.reference_level={ref!r} is not a level of "
                         f"{response_key!r} (levels: {levels})")
    ref = str(ref)
    test_level = [lv for lv in levels if lv != ref][0]
    contrast = [response_key, test_level, ref]
    contrast_label = f"{test_level}_vs_{ref}"
    print(f"DE contrast: positive log2FoldChange = higher in {test_level} than {ref}")

    summary = []
    for ct, sub in meta.groupby("cell_type", observed=True):
        per_group = sub.groupby(response_key, observed=True).size()
        row = {"cell_type": ct, "n_libraries": len(sub),
               **{f"n_{g}": int(n) for g, n in per_group.items()}}
        if len(per_group) < 2 or (per_group < min_samples).any():
            summary.append({**row, "status": "skipped_insufficient_replicates",
                            "n_sig": np.nan})
            continue
        counts = pb.loc[sub.index]
        counts = counts.loc[:, counts.sum(axis=0) > 0]
        dds = DeseqDataSet(counts=counts, metadata=sub[[response_key]].copy(),
                           design=f"~{response_key}", refit_cooks=True,
                           inference=inference, quiet=True)
        dds.deseq2()
        stat = DeseqStats(dds, contrast=contrast, alpha=alpha,
                          inference=inference, quiet=True)
        stat.summary()
        res = stat.results_df.sort_values("padj")
        safe = str(ct).replace("/", "_").replace(" ", "_")
        res.to_csv(outdir / f"pseudobulk_de_{safe}.csv")
        summary.append({**row, "status": "tested", "contrast": contrast_label,
                        "n_sig": int((res["padj"] < alpha).sum()),
                        "n_up_in_" + test_level: int(
                            ((res["padj"] < alpha) & (res["log2FoldChange"] > 0)).sum()),
                        "n_up_in_" + ref: int(
                            ((res["padj"] < alpha) & (res["log2FoldChange"] < 0)).sum())})
    out = pd.DataFrame(summary)
    out.attrs["contrast_label"] = contrast_label
    return out
