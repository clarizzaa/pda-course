#!/usr/bin/env python
"""03 — annotate clusters from marker panels and score interpretation signatures.

in : data/processed/02_clustered.h5ad
out: data/processed/03_annotated.h5ad
     results/cluster_annotation_scores.csv, results/cluster_markers.csv
     figures/02_umap_panels.png

The automatic labels are a first pass. Read results/cluster_annotation_scores.csv:
any cluster with a small margin_to_runner_up should be adjudicated by hand, the call
recorded in reports/DECISIONS.md, and set in configs/manual_labels.yaml if overridden.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scanpy as sc                                        # noqa: E402
import yaml                                                # noqa: E402

from src import annotate, plotting                         # noqa: E402
from src.config import REPO_ROOT, load_config, outdirs, set_seed, write_runlog  # noqa: E402


def main() -> None:
    cfg = load_config("scrna")
    set_seed(cfg)
    dirs = outdirs(cfg)

    adata = sc.read_h5ad(dirs["processed"] / "02_clustered.h5ad")
    cluster_key = adata.uns["pipeline"]["cluster_key"]

    panel_cols = annotate.score_panels(adata, cfg.req("markers"), prefix="score_")
    scores = annotate.label_clusters(adata, cluster_key, panel_cols, prefix="score_")
    scores.to_csv(dirs["results"] / "cluster_annotation_scores.csv", index=False)

    manual_path = REPO_ROOT / "configs" / "manual_labels.yaml"
    if manual_path.exists():
        with manual_path.open() as fh:
            mapping = yaml.safe_load(fh) or {}
        mapping = {str(k): str(v) for k, v in mapping.items()}
        if mapping:
            annotate.apply_manual_labels(adata, mapping, cluster_key)
            print(f"applied {len(mapping)} manual labels from {manual_path.name}")

    annotate.cluster_markers(adata, cluster_key).to_csv(
        dirs["results"] / "cluster_markers.csv", index=False)

    sig_cols = annotate.score_panels(adata, cfg.req("signatures"), prefix="sig_")
    adata.uns["pipeline"]["signature_cols"] = sig_cols

    plotting.umap_panels(adata, cfg, cluster_key, dirs["figures"] / "02_umap_panels.png")

    margin = float(cfg.req("annotation.min_margin"))
    ambiguous = scores.loc[scores["margin_to_runner_up"] < margin, cluster_key].astype(str).tolist()
    if ambiguous:
        print(f"[warn] clusters whose top two panels differ by <{margin}: {ambiguous} "
              "— adjudicate these by hand before trusting the labels")

    out = dirs["processed"] / "03_annotated.h5ad"
    adata.write_h5ad(out)
    write_runlog(cfg, "03_annotate",
                 {"cell_types": sorted(adata.obs["cell_type"].astype(str).unique()),
                  "ambiguous_clusters": ambiguous,
                  "manual_labels_applied": manual_path.exists()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
