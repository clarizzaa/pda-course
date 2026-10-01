"""End-to-end smoke test for src/.

The fixture below is RANDOM SYNTHETIC COUNTS. It exists only to prove the code runs and
the shapes/contracts hold; it is not data and no result from it belongs in reports/.
Real inputs live in data/raw and are declared in data/metadata/SOURCES.md.

    pytest -q tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import annotate, data_io, preprocess, qc, stats          # noqa: E402
from src.config import load_config                               # noqa: E402

N_PER_SAMPLE = 120
N_FILLER_GENES = 400


@pytest.fixture(scope="module")
def cfg():
    """The real config, with sizes shrunk to fit a toy matrix."""
    c = load_config("scrna")
    c["min_genes"] = 50
    c["n_pcs"] = 20
    c["norm"]["n_hvg"] = 150
    c["qc"]["min_cells_per_sample"] = 10
    c["keys"]["sample"] = "sample_id"
    c["keys"]["response"] = "response"
    c["keys"]["patient"] = "patient"
    c["keys"]["timepoint"] = "timepoint"
    c["keys"]["batch"] = "batch"
    c["embed"]["leiden_resolutions"] = [0.4, 0.8]
    c["embed"]["leiden_primary"] = 0.8
    c["de"]["min_samples_per_group"] = 3
    return c


@pytest.fixture(scope="module")
def toy(cfg):
    """Six samples, three latent cell populations, marker genes genuinely up in each."""
    rng = np.random.default_rng(int(cfg["random_seed"]))
    panels = {k: list(v) for k, v in cfg["markers"].items()}
    sig_genes = sorted({g for v in cfg["signatures"].values() for g in v})
    marker_genes = sorted({g for v in panels.values() for g in v} | set(sig_genes))
    filler = [f"GENE{i:04d}" for i in range(N_FILLER_GENES)]
    mito = [f"MT-ND{i}" for i in range(1, 7)]
    genes = marker_genes + filler + mito
    index = {g: i for i, g in enumerate(genes)}

    # CD8_T is included deliberately: its panel overlaps CD4_T on CD3D/CD3E, so
    # recovering it checks that the labelling rule compares panels on raw scores.
    populations = ["CD8_T", "B", "Macrophage", "Fibroblast"]
    samples = [f"S{i}" for i in range(1, 7)]
    response = dict(zip(samples, ["R", "R", "R", "NR", "NR", "NR"]))

    obs_rows, blocks = [], []
    for samp in samples:
        for cell in range(N_PER_SAMPLE):
            pop = populations[cell % len(populations)]
            lam = np.full(len(genes), 0.35)
            for g in panels[pop]:
                lam[index[g]] = 6.0
            for g in mito:
                lam[index[g]] = 1.2
            blocks.append(rng.poisson(lam))
            obs_rows.append({
                "sample_id": samp,
                "response": response[samp],
                "patient": f"P{samp[-1]}",
                "timepoint": "pre" if int(samp[-1]) % 2 else "on",
                # batch is deliberately CROSSED with response (b1 = S1,S2,S4) so the
                # confounding check has something separable to look at
                "batch": "b1" if samp in ("S1", "S2", "S4") else "b2",
                "truth": pop,
            })
    X = sparse.csr_matrix(np.vstack(blocks).astype(np.float32))
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"cell{i}" for i in range(len(obs))]
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    adata.var_names_make_unique()
    return adata


def test_assert_raw_counts_rejects_normalised(toy):
    norm = toy.copy()
    norm.X = norm.X.multiply(0.137)
    with pytest.raises(ValueError, match="raw counts"):
        data_io.assert_raw_counts(norm)
    data_io.assert_raw_counts(toy)          # the integer version passes


def test_check_design_and_confounding(toy, cfg):
    design = data_io.check_design(toy, cfg)
    assert len(design) == 6
    assert design["n_cells"].sum() == toy.n_obs
    conf = data_io.check_confounding(design, cfg)
    assert conf.attrs["fully_nested"] is False   # b1/b2 each span both groups


def test_confounding_check_flags_a_nested_batch(toy, cfg):
    """A batch perfectly nested in response must be flagged, not silently passed."""
    nested = toy.copy()
    nested.obs["batch"] = np.where(
        nested.obs["response"].astype(str) == "R", "b1", "b2")
    design = data_io.check_design(nested, cfg)
    conf = data_io.check_confounding(design, cfg)
    assert conf.attrs["fully_nested"] is True


def test_check_design_rejects_missing_labels(toy, cfg):
    broken = toy.copy()
    broken.obs["response"] = broken.obs["response"].astype(object)
    broken.obs.loc[broken.obs_names[:5], "response"] = np.nan
    with pytest.raises(ValueError, match="carry no"):
        data_io.check_design(broken, cfg)


@pytest.fixture(scope="module")
def processed(toy, cfg):
    adata = toy.copy()
    qc.annotate_qc(adata, cfg)
    assert adata.var["mt"].sum() == 6
    filtered, audit = qc.filter_cells(adata, cfg)
    assert audit["kept"].sum() == filtered.n_obs
    assert filtered.n_obs > 0.5 * toy.n_obs

    preprocess.normalise(filtered, cfg)
    assert "counts" in filtered.layers
    preprocess.select_hvgs(filtered, cfg)
    rep = preprocess.embed(filtered, cfg, int(cfg["random_seed"]))
    assert rep == "X_pca_harmony"
    cluster_key = preprocess.cluster(filtered, cfg, rep, int(cfg["random_seed"]))

    panel_cols = annotate.score_panels(filtered, cfg["markers"], prefix="score_")
    annotate.label_clusters(filtered, cluster_key, panel_cols, prefix="score_")
    sig_cols = annotate.score_panels(filtered, cfg["signatures"], prefix="sig_")
    return filtered, cluster_key, sig_cols


def test_counts_layer_is_untouched_by_normalisation(processed, toy):
    adata = processed[0]
    assert adata.layers["counts"].max() == toy[adata.obs_names, adata.var_names].X.max()


def test_labels_recover_the_planted_populations(processed):
    adata = processed[0]
    assigned = set(adata.obs["cell_type"].astype(str).unique())
    planted = set(adata.obs["truth"].unique())
    assert planted <= assigned, f"missed {planted - assigned}"
    agree = (adata.obs["cell_type"].astype(str) == adata.obs["truth"]).mean()
    assert agree > 0.8, f"label agreement only {agree:.2f}"


def test_composition_is_per_sample(processed, cfg):
    adata = processed[0]
    long = stats.celltype_fractions(adata, cfg)
    sums = long.groupby("sample_id", observed=True)["fraction"].sum()
    assert np.allclose(sums.values, 1.0)
    res = stats.composition_response_test(long, cfg)
    assert {"p", "p_adj", "status"} <= set(res.columns)
    assert (res["n_R"] <= 3).all() and (res["n_NR"] <= 3).all()   # samples, not cells
    paired = stats.composition_paired_test(long, cfg)
    assert paired.empty or "n_patients" in paired.columns


def test_signature_test_aggregates_to_sample(processed, cfg):
    adata, _, sig_cols = processed
    res = stats.signature_response_test(adata, cfg, sig_cols, "CD8_T")
    if not res.empty:
        assert res["n_R"].max() <= 3 and res["n_NR"].max() <= 3


def test_pseudobulk_sums_counts_and_fits(processed, cfg, tmp_path):
    adata = processed[0]
    pb, meta = stats.make_pseudobulk(adata, cfg)
    assert pb.index.equals(meta.index)
    assert (pb.values >= 0).all()
    assert pb.dtypes.unique().tolist() == [np.dtype("int64")]
    # a library must equal the column sum of its member cells
    lib = meta.index[0]
    samp, ct = lib.split("||")
    mask = ((adata.obs[cfg["keys"]["sample"]].astype(str) == samp)
            & (adata.obs["cell_type"].astype(str) == ct))
    expected = np.asarray(adata.layers["counts"][mask.values].sum(axis=0)).ravel()
    assert np.allclose(pb.loc[lib].values, expected)

    sample_meta = adata.obs.groupby(cfg["keys"]["sample"], observed=True).first()
    cfg["de"]["reference_level"] = "NR"
    summary = stats.deseq2_by_celltype(pb, meta, sample_meta, cfg, tmp_path)
    assert not summary.empty
    assert set(summary["status"]) <= {"tested", "skipped_insufficient_replicates"}
    # the contrast direction must be explicit and recorded, not implied
    assert summary.attrs["contrast_label"] == "R_vs_NR"
    tested = summary[summary["status"] == "tested"]
    if not tested.empty:
        assert (tested["contrast"] == "R_vs_NR").all()
    for ct in summary.loc[summary["status"] == "tested", "cell_type"]:
        safe = str(ct).replace("/", "_").replace(" ", "_")
        res = pd.read_csv(tmp_path / f"pseudobulk_de_{safe}.csv", index_col=0)
        assert {"log2FoldChange", "pvalue", "padj"} <= set(res.columns)


def test_de_skips_underpowered_celltypes(processed, cfg, tmp_path):
    """With min_samples_per_group above the design, everything must be skipped."""
    adata = processed[0]
    pb, meta = stats.make_pseudobulk(adata, cfg)
    strict = load_config("scrna")
    strict["keys"] = cfg["keys"]
    strict["de"]["min_samples_per_group"] = 99
    sample_meta = adata.obs.groupby(cfg["keys"]["sample"], observed=True).first()
    summary = stats.deseq2_by_celltype(pb, meta, sample_meta, strict, tmp_path)
    assert (summary["status"] == "skipped_insufficient_replicates").all()
    assert summary["n_sig"].isna().all()
