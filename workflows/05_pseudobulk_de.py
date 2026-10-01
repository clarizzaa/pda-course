#!/usr/bin/env python
"""05 — pseudobulk differential expression, responder vs non-responder, per cell type.

in : data/processed/03_annotated.h5ad
out: results/pseudobulk_counts.csv, results/pseudobulk_libraries.csv,
     results/pseudobulk_de_summary.csv, results/pseudobulk_de_<cell_type>.csv
     figures/05_volcano_<cell_type>.png

Raw counts are summed within (sample x cell type) and the contrast is fitted across
samples with PyDESeq2. The alternative — a per-cell Wilcoxon test — treats thousands of
cells from one patient as independent replicates and inflates significance by orders of
magnitude. The unit of replication is the patient (WORKSPACE.md rule 6).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                        # noqa: E402
import scanpy as sc                                        # noqa: E402

from src import plotting, stats                            # noqa: E402
from src.config import load_config, outdirs, set_seed, write_runlog  # noqa: E402


def main() -> None:
    cfg = load_config("scrna")
    set_seed(cfg)
    dirs = outdirs(cfg)

    adata = sc.read_h5ad(dirs["processed"] / "03_annotated.h5ad")
    pb, pb_meta = stats.make_pseudobulk(adata, cfg)
    pb.to_csv(dirs["results"] / "pseudobulk_counts.csv")
    pb_meta.to_csv(dirs["results"] / "pseudobulk_libraries.csv")
    print(f"{pb.shape[0]} pseudobulk libraries x {pb.shape[1]} genes")

    sample_meta = adata.obs.groupby(cfg.key("sample"), observed=True).first()
    summary = stats.deseq2_by_celltype(pb, pb_meta, sample_meta, cfg, dirs["results"])
    summary.to_csv(dirs["results"] / "pseudobulk_de_summary.csv", index=False)
    print(summary.to_string(index=False))

    contrast_label = summary.attrs.get("contrast_label", "")
    for ct in summary.loc[summary["status"] == "tested", "cell_type"]:
        safe = str(ct).replace("/", "_").replace(" ", "_")
        res = pd.read_csv(dirs["results"] / f"pseudobulk_de_{safe}.csv", index_col=0)
        plotting.de_volcano(res, str(ct), cfg, dirs["figures"] / f"05_volcano_{safe}.png",
                            contrast_label=contrast_label)

    write_runlog(cfg, "05_pseudobulk_de",
                 {"n_libraries": int(pb.shape[0]),
                  "tested": summary.loc[summary["status"] == "tested",
                                        "cell_type"].astype(str).tolist(),
                  "skipped": summary.loc[summary["status"] != "tested",
                                         "cell_type"].astype(str).tolist()})


if __name__ == "__main__":
    main()
