# Scanpy randomized PCA benchmark modules

Scanpy/sklearn modules used in the randomized PCA OmniBenchmark pipeline.

## Upstream and modifications

This repository is based on the `omni-scrna/scanpy` OmniBenchmark module.

Original repository:
https://github.com/omnibenchmark/omni-scrna/tree/main/scanpy

The code was adapted for this project to support randomized PCA experiments, including repeated random seeds, configurable randomized-SVD parameters, PCA loadings/scores export, and downstream clustering of selected benchmark runs.
## PCA

`pca.py` computes either exact or randomized PCA on the preprocessed
cell × gene matrix supplied by the benchmark.

### Exact PCA

Exact PCA is computed through Scanpy.

### Randomized PCA

Randomized PCA uses `sklearn.decomposition.PCA` with
`svd_solver="randomized"`.

The randomized approximation is controlled by:

- `random_seed`
- `n_iter`
- `n_oversamples`

All runs compute 50 principal components unless configured otherwise.

### Input

The PCA module receives the normalized, HVG-selected matrix produced by
the upstream benchmark stages.

### Outputs

- `{name}_pcas.tsv`: PCA scores, cells × PCs
- `{name}_loadings.tsv`: PCA loadings, genes × PCs

## kNN and clustering

`cluster.py` performs Leiden clustering from a precomputed neighbor graph.

`selected_clustering.py` applies the clustering module to the exact,
best-seed and worst-seed graphs selected by the benchmark and produces a
manifest of clustering outputs.

The Leiden random seed is kept fixed when comparing PCA runs so that
differences reflect changes propagated from PCA rather than additional
clustering randomness.

## Reproducibility

The OmniBenchmark environment is defined in:

`envs/scanpy.yaml`

Benchmark plans should reference a specific Git commit of this repository
rather than a moving branch.