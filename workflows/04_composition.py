#!/usr/bin/env python
"""04 — cell-type composition and signature scores, responder vs non-responder.

in : data/processed/03_annotated.h5ad
out: results/celltype_composition.csv, results/composition_response_test.csv,
     results/composition_paired_timepoint.csv (only if keys.patient + keys.timepoint set),
     results/signature_response_test.csv
     figures/03_composition_response.png, figures/04_signature_scores.png

Every test here is across samples, not across cells (WORKSPACE.md rule 6).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scanpy as sc                                        # noqa: E402

from src import plotting, stats                            # noqa: E402
from src.config import load_config, outdirs, set_seed, write_runlog  # noqa: E402


def main() -> None:
    cfg = load_config("scrna")
    set_seed(cfg)
    dirs = outdirs(cfg)

    adata = sc.read_h5ad(dirs["processed"] / "03_annotated.h5ad")
    sig_cols = adata.uns["pipeline"]["signature_cols"]

    long = stats.celltype_fractions(adata, cfg)
    long.to_csv(dirs["results"] / "celltype_composition.csv", index=False)

    comp = stats.composition_response_test(long, cfg)
    comp.to_csv(dirs["results"] / "composition_response_test.csv", index=False)
    print(comp.to_string(index=False))

    paired = stats.composition_paired_test(long, cfg)
    if paired.empty:
        print("[note] paired pre/on-treatment test not assessed "
              "(keys.patient / keys.timepoint not set, or no usable pairs)")
    else:
        paired.to_csv(dirs["results"] / "composition_paired_timepoint.csv", index=False)

    compartment = str(cfg.req("signature_compartment"))
    sig = stats.signature_response_test(adata, cfg, sig_cols, compartment)
    if sig.empty:
        print(f"[note] signature test not assessed: no {compartment!r} cells, "
              "or too few samples per group")
    else:
        sig.to_csv(dirs["results"] / "signature_response_test.csv", index=False)
        print(sig.to_string(index=False))

    plotting.composition_box(long, cfg, dirs["figures"] / "03_composition_response.png")
    plotting.signature_violins(adata, cfg, sig_cols, compartment,
                               dirs["figures"] / "04_signature_scores.png")

    alpha = float(cfg.req("composition.alpha"))
    hits = comp.loc[comp["p_adj"] < alpha, "cell_type"].tolist() if "p_adj" in comp else []
    write_runlog(cfg, "04_composition",
                 {"n_celltypes": int(long["cell_type"].nunique()),
                  "n_samples": int(long[cfg.key("sample")].nunique()),
                  "composition_hits_fdr": hits,
                  "paired_test_run": not paired.empty})


if __name__ == "__main__":
    main()
