# Analysis decisions

Running log of every analysis decision and why (WORKSPACE.md rule 3). Newest last.
Entries that record a *result* must cite the file in `results/` or `figures/` that
supports it (rule 7). No dataset has been analysed yet, so nothing below cites results.

---

## Setup

### Experimental unit is the sample, not the cell
Every response comparison aggregates to the sample before testing: cell-type fractions
per sample (`src/stats.celltype_fractions`), per-sample mean signature scores
(`signature_response_test`), and pseudobulk count libraries for DE (`make_pseudobulk`).
A per-cell test treats thousands of cells from one patient as independent replicates;
with ~6 samples the nominal n would be inflated by three orders of magnitude and almost
every gene would clear any threshold. The per-cell Wilcoxon test that *is* run
(`annotate.cluster_markers`) is used only to name clusters, never to compare groups.

### QC thresholds, and why the upper bound is per-sample
`min_genes` and `max_mito_fraction` come from `configs/scrna.yaml` and are applied
globally. The upper `n_genes` bound (a crude doublet proxy) is instead the per-sample
98th percentile: one global upper cutoff would remove a different fraction of each
sample depending on its sequencing depth, and since depth varies by sample, that
differential cell loss would appear downstream as a composition difference between
response groups. `results/qc_filter_audit.csv` records which rule removed what, per
sample, so the loss is auditable.

### Integration corrects the embedding only
Harmony is run on the PCA embedding and written to `obsm["X_pca_harmony"]`, which is
what clustering uses. The expression matrix is left uncorrected and all differential
testing reads from it. Patient identity is confounded with response in this design
(each patient contributes to exactly one response group), so correcting expression on
the sample key would remove part of the effect the analysis is trying to measure.

### Clonotype genes excluded from HVG selection
`norm.exclude_hvg_pattern` drops TR[ABGD][VJC] and IG[HKL][VJC] V/J/C segments and
mitochondrial genes from the HVG set. A single expanded T-cell clone makes its own V
genes highly variable across the dataset, which produces clusters that track that one
clone rather than a cell state.

### Cluster labels come from raw panel scores, not z-scores across clusters
`annotate.label_clusters` picks each cluster's label by argmax across marker panels on
the raw `score_genes` values, and keeps the across-cluster z-scores only as `z_*`
diagnostic columns. The first implementation argmaxed the z-scores and mislabelled the
CD8 T population: CD8_T (CD3D/CD3E/CD8A/CD8B) and CD4_T (CD3D/CD3E/CD4/IL7R) share half
their genes, so a CD8 cluster elevates both panels. On raw scores CD8_T wins — 4/4 genes
up against 2/4 — but z-scoring each panel across clusters normalises away that magnitude
difference, leaving both panels looking equally "high in one cluster" and the pick
essentially arbitrary. Caught by `tests/test_pipeline_smoke.py::
test_labels_recover_the_planted_populations`. Labels remain a first pass: clusters whose
top two panels differ by less than `annotation.min_margin` are printed for manual
adjudication, and overrides belong in `configs/manual_labels.yaml` with the reason
recorded here.

### The DE contrast is stated explicitly
`de.reference_level` names the baseline level of the response variable and
`deseq2_by_celltype` builds an explicit `contrast=[key, test, ref]`. Leaving the
direction to the library's level ordering would make the sign of every log2 fold change
depend on alphabetical accident, which would silently invert the biological
interpretation. The contrast label is recorded in
`results/pseudobulk_de_summary.csv` and printed on the volcano axis.

### Under-powered cell types are skipped, not tested
A cell type without `de.min_samples_per_group` pseudobulk libraries in both groups is
recorded as `skipped_insufficient_replicates` rather than fitted. A fit on 2-vs-2
libraries that returns no significant genes reads as evidence of no difference when it
is really absence of power, and that is the more misleading of the two outcomes.

### Composition tests are screening, not confirmatory
Mann-Whitney per cell type on per-sample fractions, BH-corrected across cell types.
Fractions sum to 1, so cell types are not independent and a genuine expansion of one
compartment mechanically deflates all others. A Dirichlet-multinomial model (scCODA) is
the correct confirmatory step if the assignment calls for one; it is not implemented.

### harmonypy called directly rather than through scanpy's wrapper
`preprocess.run_harmony` calls `harmonypy.run_harmony` and checks the orientation of
`Z_corr` before assigning it. `scanpy.external.pp.harmony_integrate` transposes the
result, which was correct for harmonypy <2 (it returned `(dims, cells)`) but is wrong
for harmonypy >=2 (which returns `(cells, dims)`); with scanpy 1.12.4 and harmonypy
2.0.2 the wrapper hands anndata a transposed array and the assignment raises. The
explicit shape check works with either version.

### pydeseq2 runs serially
`de.n_cpus: 1`. Parallel inference fans out over joblib/loky, which needs POSIX
semaphores and raises `PermissionError: [Errno 1]` on `SC_SEM_NSEMS_MAX` in sandboxed
environments. The pseudobulk matrices are small enough that serial fitting is not a
constraint.

### Not assessed
Doublet detection (Scrublet), ambient-RNA correction, CD8 subclustering and pseudotime,
cell-cell communication, and TCR clonality are not implemented. Clustering-assumption
and dispersion diagnostics beyond what pydeseq2 prints are not examined.

---

## Analysis

_Add one entry per decision taken once real data is in `data/raw`. Record the cluster
labels adjudicated by hand, the resolution chosen and why, and any sample excluded —
each citing the file in `results/` or `figures/` that supports it._
