#!/usr/bin/env python
"""02 — normalise, select HVGs, embed (PCA + Harmony), cluster (Leiden sweep).

in : data/processed/01_filtered.h5ad
out: data/processed/02_clustered.h5ad
     results/leiden_resolution_sweep.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scanpy as sc                                        # noqa: E402

from src import preprocess                                 # noqa: E402
from src.config import load_config, outdirs, set_seed, write_runlog  # noqa: E402


def main() -> None:
    cfg = load_config("scrna")
    seed = set_seed(cfg)
    dirs = outdirs(cfg)

    adata = sc.read_h5ad(dirs["processed"] / "01_filtered.h5ad")
    preprocess.normalise(adata, cfg)
    n_hvg = preprocess.select_hvgs(adata, cfg)
    rep = preprocess.embed(adata, cfg, seed)
    cluster_key = preprocess.cluster(adata, cfg, rep, seed)

    sweep = preprocess.resolution_sweep_table(adata, cfg)
    sweep.to_csv(dirs["results"] / "leiden_resolution_sweep.csv", index=False)
    print(sweep.to_string(index=False))

    adata.uns["pipeline"] = {"cluster_key": cluster_key, "embedding_rep": rep}
    out = dirs["processed"] / "02_clustered.h5ad"
    adata.write_h5ad(out)
    write_runlog(cfg, "02_normalize_cluster",
                 {"n_hvg_selected": n_hvg, "embedding_rep": rep,
                  "cluster_key": cluster_key,
                  "n_clusters": int(adata.obs[cluster_key].nunique())})
    print(f"wrote {out} (clusters in obs[{cluster_key!r}], embedding {rep})")


if __name__ == "__main__":
    main()
