#!/usr/bin/env python
"""01 — load data/raw, validate the design, compute QC metrics, filter cells.

in : configs/scrna.yaml -> paths.matrix (+ optional paths.sample_meta)
out: data/processed/01_filtered.h5ad
     results/design.csv, results/qc_summary.csv, results/qc_filter_audit.csv,
     results/confounding_response_by_batch.csv (if keys.batch is set)
     figures/01_qc_per_sample.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_io, plotting, qc                      # noqa: E402
from src.config import load_config, outdirs, set_seed, write_runlog  # noqa: E402


def main() -> None:
    cfg = load_config("scrna")
    set_seed(cfg)
    dirs = outdirs(cfg)

    matrix = cfg.path("paths.matrix")
    adata = data_io.load_matrix(matrix)
    data_io.assert_raw_counts(adata)
    print(f"loaded {adata.n_obs} cells x {adata.n_vars} genes from {matrix.name}")

    meta_path = cfg.opt("paths.sample_meta")
    data_io.attach_sample_meta(
        adata, Path(cfg.path("paths.sample_meta")) if meta_path else None, cfg.key("sample")
    )

    design = data_io.check_design(adata, cfg)
    design.to_csv(dirs["results"] / "design.csv", index=False)
    conf = data_io.check_confounding(design, cfg)
    if not conf.empty:
        conf.to_csv(dirs["results"] / "confounding_response_by_batch.csv")

    qc.annotate_qc(adata, cfg)
    pre = qc.qc_table(adata, cfg.key("sample"), "pre_filter")
    plotting.qc_violins(adata, cfg, dirs["figures"] / "01_qc_per_sample.png")

    filtered, audit = qc.filter_cells(adata, cfg)
    post = qc.qc_table(filtered, cfg.key("sample"), "post_filter")

    import pandas as pd
    pd.concat([pre, post]).to_csv(dirs["results"] / "qc_summary.csv", index=False)
    audit.to_csv(dirs["results"] / "qc_filter_audit.csv", index=False)

    out = dirs["processed"] / "01_filtered.h5ad"
    filtered.write_h5ad(out)
    write_runlog(cfg, "01_qc", {"n_cells_in": int(adata.n_obs),
                                "n_cells_out": int(filtered.n_obs),
                                "n_genes_out": int(filtered.n_vars),
                                "n_samples_out": int(filtered.obs[cfg.key("sample")].nunique())})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
