from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, svds


# ==================================================
# SETTINGS
# ==================================================

K = 50

N_ITERS = [2, 3, 4, 5]
N_OVERSAMPLES = [10, 20, 30, 40]
SEEDS = range(1, 51)


# ==================================================
# PATHS
# ==================================================

MATRIX_FILE = Path(
    "test_results/feat_select/"
    "p38_s42_strong_signal_normalized_selected.h5"
)

EXACT_SCORES_FILE = Path(
    "test_results/pca_full/"
    "p38_s42_strong_signal_full_pcas.tsv"
)

EXACT_LOADINGS_FILE = Path(
    "test_results/pca_full/"
    "p38_s42_strong_signal_full_loadings.tsv"
)

# Parent directory containing:
# n2_over10/, n2_over20/, ..., n5_over40/
RANDOM_ROOT = Path(
    "test_results/pca_randomized/38_strong_signal"
)

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FIGURE = (
    OUTPUT_DIR / "strong_signal_accuracy_heatmap.png"
)

OUTPUT_CSV = (
    OUTPUT_DIR / "strong_signal_accuracy_heatmap.csv"
)


# ==================================================
# READ MATRIX
# ==================================================

def read_matrix(path):

    with h5py.File(path, "r") as h5:

        g = h5["matrix"]

        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(g["shape"][:])

    # Stored as genes x cells
    X = csc_matrix(
        (data, indices, indptr),
        shape=shape
    ).T.toarray()

    X = X.astype(np.float64)

    # PCA is performed on mean-centred data
    X -= X.mean(axis=0, keepdims=True)

    return X


# ==================================================
# READ PCA OUTPUT
# ==================================================

def read_pca(path):

    df = pd.read_csv(
        path,
        sep="\t"
    )

    # First column = cell_id or gene_id
    return df.iloc[:, 1:].to_numpy(
        dtype=np.float64
    )


# ==================================================
# FIND RANDOMIZED OUTPUT FILE
# ==================================================

def find_output(folder, seed, n_iter, n_over, kind):

    pattern = (
        f"*seed{seed}_"
        f"iter{n_iter}_"
        f"over{n_over}_"
        f"{kind}.tsv"
    )

    files = list(folder.glob(pattern))

    if len(files) == 0:
        return None

    if len(files) > 1:
        raise RuntimeError(
            f"More than one file matched {pattern} "
            f"in {folder}"
        )

    return files[0]


# ==================================================
# RESIDUAL SPECTRAL NORM
# ==================================================

def residual_spectral_norm(
    X,
    scores,
    loadings
):

    """
    Compute

        ||X - scores @ loadings.T||_2

    The spectral norm is the largest singular
    value of the residual.
    """

    n_cells, n_genes = X.shape

    def matvec(v):

        return (
            X @ v
            - scores @ (loadings.T @ v)
        )

    def rmatvec(v):

        return (
            X.T @ v
            - loadings @ (scores.T @ v)
        )

    residual = LinearOperator(
        shape=(n_cells, n_genes),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64
    )

    sigma_max = svds(
        residual,
        k=1,
        which="LM",
        return_singular_vectors=False,
        tol=1e-6
    )[0]

    return float(sigma_max)


# ==================================================
# EXACT PCA REFERENCE
# ==================================================

print("Loading matrix...")

X = read_matrix(MATRIX_FILE)

print("Matrix:", X.shape)


exact_scores = read_pca(
    EXACT_SCORES_FILE
)[:, :K]

exact_loadings = read_pca(
    EXACT_LOADINGS_FILE
)[:, :K]


print("Computing exact rank-50 residual...")

exact_error = residual_spectral_norm(
    X,
    exact_scores,
    exact_loadings
)

print(
    f"Exact residual norm = "
    f"{exact_error:.6f}"
)


# ==================================================
# PARAMETER GRID
# ==================================================

median_quality = np.full(
    (
        len(N_ITERS),
        len(N_OVERSAMPLES)
    ),
    np.nan
)

all_results = []


for i, n_iter in enumerate(N_ITERS):

    for j, n_over in enumerate(N_OVERSAMPLES):

        # Important:
        # results are inside separate folders
        run_folder = RANDOM_ROOT / (
            f"n{n_iter}_over{n_over}"
        )

        print(
            f"\n===================================="
        )
        print(
            f"iter={n_iter}, "
            f"oversamples={n_over}"
        )
        print(
            f"folder: {run_folder}"
        )

        qualities = []


        for seed in SEEDS:

            scores_file = find_output(
                run_folder,
                seed,
                n_iter,
                n_over,
                "pcas"
            )

            loadings_file = find_output(
                run_folder,
                seed,
                n_iter,
                n_over,
                "loadings"
            )


            if (
                scores_file is None
                or loadings_file is None
            ):

                print(
                    f"WARNING: missing seed {seed}"
                )

                continue


            scores = read_pca(
                scores_file
            )[:, :K]

            loadings = read_pca(
                loadings_file
            )[:, :K]


            randomized_error = (
                residual_spectral_norm(
                    X,
                    scores,
                    loadings
                )
            )


            # --------------------------------------
            # Approximation quality
            #
            # 1.0 = exact PCA optimum
            #
            # smaller values = worse approximation
            # --------------------------------------

            quality = (
                exact_error
                / randomized_error
            )

            qualities.append(quality)


            all_results.append(
                {
                    "n_iter": n_iter,
                    "n_oversamples": n_over,
                    "seed": seed,
                    "randomized_error":
                        randomized_error,
                    "quality": quality,
                }
            )


        if len(qualities) > 0:

            median_quality[i, j] = (
                np.median(qualities)
            )

            print(
                f"Runs found: {len(qualities)}"
            )

            print(
                f"Median quality: "
                f"{median_quality[i, j]:.6f}"
            )


# ==================================================
# SAVE NUMERICAL RESULTS
# ==================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_DIR
    / "strong_signal_accuracy_all_seeds.csv",
    index=False
)


heatmap_df = pd.DataFrame(
    median_quality,
    index=N_ITERS,
    columns=N_OVERSAMPLES
)

heatmap_df.index.name = "n_iter"
heatmap_df.columns.name = "n_oversamples"

heatmap_df.to_csv(
    OUTPUT_CSV
)


print("\nMedian approximation quality:")
print(heatmap_df)


# ==================================================
# HEATMAP
# ==================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


im = ax.imshow(
    median_quality,
    cmap="viridis",
    aspect="auto",
    vmin=np.nanmin(median_quality),
    vmax=1.0
)


# --------------------------------------------------
# AXES
# --------------------------------------------------

ax.set_xticks(
    np.arange(len(N_OVERSAMPLES))
)

ax.set_xticklabels(
    N_OVERSAMPLES
)

ax.set_yticks(
    np.arange(len(N_ITERS))
)

ax.set_yticklabels(
    N_ITERS
)

ax.set_xlabel(
    "Number of oversamples"
)

ax.set_ylabel(
    "Number of power iterations"
)

ax.set_title(
    "strong signal — randomized PCA approximation quality\n"
    "Median across 50 seeds, rank = 50"
)


# ==================================================
# VALUES INSIDE CELLS
# ==================================================

for i in range(len(N_ITERS)):

    for j in range(len(N_OVERSAMPLES)):

        value = median_quality[i, j]

        if np.isnan(value):
            text = "NA"
        else:
            text = f"{value:.4f}"

        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=11
        )


# ==================================================
# COLOR GRADIENT / LEGEND
# ==================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.03
)

cbar.set_label(
    "Median approximation quality\n"
    "(1.0 = exact PCA optimum)"
)


plt.tight_layout()


# ==================================================
# SAVE
# ==================================================

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "\nSaved figure:",
    OUTPUT_FIGURE
)

print(
    "Saved heatmap values:",
    OUTPUT_CSV
)