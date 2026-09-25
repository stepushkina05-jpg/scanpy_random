#!/usr/bin/env python3
"""
PCA module for the randomized-PCA benchmark.

Input
-----
Preprocessed cell × gene matrix from the FEAT stage
(`--normalized_selected_h5`).

Outputs
-------
{name}_pcas.tsv
    Cell × PC score matrix.

{name}_loadings.tsv
    Gene × PC loading matrix.

Methods
-------
Exact PCA:
    Scanpy PCA using the requested exact solver.

Randomized PCA:
    sklearn.decomposition.PCA with svd_solver="randomized".
    The approximation is controlled by:
        --random_seed
        --n_iter
        --n_oversamples

Preprocessing assumptions
-------------------------
The input matrix is already normalized and restricted to selected HVGs.
Genes are mean-centered for PCA but are not variance-scaled.

The randomized sklearn implementation currently operates on the dense
matrix representation when --dense true is used.
"""


import argparse
import os
import sys
from pathlib import Path

from sklearn.decomposition import PCA
import h5py
import numpy as np
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local writers
from common import cli  # noqa: E402
from writers import Embedding, Loadings, write_embeddings, write_loadings  # noqa: E402
from phases import phase  # noqa: E402
from obkit.logger import init_logger  # noqa: E402


def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `PCA` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="PCA module (scanpy-backed)")
    cli.add_base_args(p)             # --output_dir, --name
    cli.add_stage_args(p, "PCA")     # --normalized_selected_h5
    p.add_argument("--solver", type=str, required=True,
                   choices=["arpack", "randomized", "full"], help="PCA solver")
    # Dense vs sparse is its own axis, not a side effect of the solver: the
    # paper contrasts the two, and only the dense path puts real work through
    # BLAS-3. Materialised in the loader so peak RSS reflects the format under
    # test instead of holding a sparse and a dense copy at once.
    p.add_argument("--dense", type=str, default="false", choices=["true", "false"],
                   help="materialise the matrix dense before PCA")
    # Snakemake exports OMP_NUM_THREADS = <rule threads> (default 1), pinning
    # the BLAS to one thread whichever implementation is linked. 0 = inherit.
    # Compute precision. The FEAT matrix arrives float64 (R numeric), so this
    # module computes in float64 today; scanpy's own dtype argument sets the
    # dtype of the *result*, not of the computation, so it cannot isolate this.
    # "input" leaves the matrix as read.
    p.add_argument("--dtype", type=str, default="input",
                   choices=["input", "float32", "float64"],
                   help="cast the matrix before PCA (input = leave as read)")
    p.add_argument("--blas_threads", type=int, default=0,
                   help="BLAS threads (0 = inherit OMP_NUM_THREADS)")
    p.add_argument("--n_components", type=int, required=True,
                   help="Number of principal components to compute")
    p.add_argument("--random_seed", type=int, required=True,
                   help="Seed for randomized solvers (and for reproducibility)")
    p.add_argument("--n_iter", type=int, default=2)
    p.add_argument("--n_oversamples", type=int, default=10)
    return p.parse_args()


def load_matrix(h5_path, dense=False):
    with h5py.File(h5_path, "r") as h5:
        g = h5["matrix"]
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(g["shape"][:])
        gene_ids = g["genes"][:].astype(str)
        cell_ids = g["barcodes"][:].astype(str)

    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()  # cells x genes
    if dense:
        # Drop the sparse copy before AnnData is built, so peak reflects one
        # representation rather than both.
        X = X.toarray()
        print(f"  dense matrix: {X.nbytes / 1e6:.0f} MB")
    adata = ad.AnnData(X=X)
    adata.obs_names = cell_ids
    adata.var_names = gene_ids
    return adata


def run_pca(adata, args):
    # Chunked mode triggers IncrementalPCA. It is most effective with backed
    # AnnData; for in-memory data it provides no memory benefit and is slower.
    # Sparse inputs get densified per chunk during partial_fit.
    #chunked = args.chunked == "true"
    chunked = False

    limits = args.blas_threads if args.blas_threads > 0 else None
    if limits:
        print(f"  blas threads limited to {limits} "
              f"(OMP_NUM_THREADS was {os.environ.get('OMP_NUM_THREADS', 'unset')})")
    with threadpool_limits(limits=limits):
      if args.solver == "randomized":
        print(f"  n_iter: {args.n_iter}")
        print(f"  n_oversamples: {args.n_oversamples}")
        pca = PCA(
            n_components=args.n_components,
            svd_solver="randomized",
            random_state=args.random_seed,
            iterated_power=args.n_iter,
            n_oversamples=args.n_oversamples,
        )

        embedding = pca.fit_transform(adata.X)
        loadings = pca.components_.T
        variance = pca.explained_variance_
        variance_ratio = pca.explained_variance_ratio_
      else:
        sc.pp.pca(
            adata,
            n_comps=args.n_components,
            zero_center=True,
            svd_solver=args.solver if not chunked else None,
            random_state=args.random_seed,
            chunked=chunked,
            chunk_size=args.chunk_size if chunked else None,
        )

        embedding = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        loadings = np.asarray(adata.varm["PCs"], dtype=np.float64)
        variance = np.asarray(adata.uns["pca"]["variance"], dtype=np.float64)
        variance_ratio = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=np.float64)
        
    embedding = np.asarray(embedding, dtype=np.float64)
    loadings = np.asarray(loadings, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    variance_ratio = np.asarray(variance_ratio, dtype=np.float64)

    return embedding, loadings, variance, variance_ratio


def validate_args(args):
    #chunked = args.chunked == "true"
    chunked = False

    if chunked:
        if args.chunk_size is None:
            sys.exit("error: --chunk_size is required when --chunked=true")
        if args.solver is not None:
            sys.exit("error: --solver must not be set when --chunked=true (IncrementalPCA ignores it)")
    else:
        if args.chunk_size is not None:
            sys.exit("error: --chunk_size must not be set when --chunked=false")
        if args.solver is None:
            sys.exit("error: --solver is required when --chunked=false")



def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "normalized_selected_h5", "solver", "n_components", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output_dir))

    with phase("load") as attrs:
        if args.solver == "full" and args.dense != "true":
            raise SystemExit("--solver full is dense-only (sklearn refuses sparse "
                             "input); pass --dense true rather than let scanpy "
                             "silently substitute another solver")
        adata = load_matrix(args.normalized_selected_h5, dense=args.dense == "true")
        if args.dtype != "input":
            adata.X = adata.X.astype(args.dtype)
        print(f"  compute dtype: {adata.X.dtype}")
        attrs["n_cells"] = adata.n_obs
        attrs["n_genes"] = adata.n_vars
    gene_ids = np.array(adata.var_names)
    cell_ids = np.array(adata.obs_names)
    print(f"  matrix (cells x genes): {adata.shape}")

    with phase("pca") as attrs:
        embedding, loadings, variance, variance_ratio = run_pca(adata, args)
        attrs["solver"] = args.solver or "chunked"
        attrs["n_components"] = args.n_components
    print(f"  embedding: {embedding.shape}, loadings: {loadings.shape}")

    with phase("write"):
        col_names = [f"PC{i + 1}" for i in range(embedding.shape[1])]
        embedding_out = Path(args.output_dir) / f"{args.name}_pcas.tsv"
        write_embeddings(Embedding(embedding, list(cell_ids), col_names), embedding_out)

        loadings_out = Path(args.output_dir) / f"{args.name}_loadings.tsv"
        write_loadings(Loadings(loadings, list(gene_ids), col_names), loadings_out)
    print(f"  wrote: {embedding_out}")
    print(f"  wrote: {loadings_out}")


if __name__ == "__main__":
    main()
