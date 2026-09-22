from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

EXACT_FILE = Path(
    "test_results/pca_full/"
    "p38_s42_strong_signal_full_pcas.tsv"
)

# Contains:
# n2_over10/
# n2_over20/
# ...
# n5_over40/
RANDOM_ROOT = Path(
    "test_results/pca_randomized/38_strong_signal"
)

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FIGURE = (
    OUTPUT_DIR / "strong_signal_correlation_heatmap.png"
)

OUTPUT_CSV = (
    OUTPUT_DIR / "strong_signal_correlation_heatmap.csv"
)

ALL_SEEDS_CSV = (
    OUTPUT_DIR / "strong_signal_correlation_all_seeds.csv"
)


# ==================================================
# READ PCA SCORES
# ==================================================

def read_pca(path):

    df = pd.read_csv(
        path,
        sep="\t"
    )

    ids = df.iloc[:, 0].astype(str).to_numpy()

    scores = df.iloc[:, 1:K + 1].to_numpy(
        dtype=np.float64
    )

    return ids, scores


# ==================================================
# FIND RANDOMIZED PCA FILE
# ==================================================

def find_pca_file(
    folder,
    seed,
    n_iter,
    n_over
):

    pattern = (
        f"*seed{seed}_"
        f"iter{n_iter}_"
        f"over{n_over}_"
        f"pcas.tsv"
    )

    files = list(
        folder.glob(pattern)
    )

    if len(files) == 0:
        return None

    if len(files) > 1:
        raise RuntimeError(
            f"More than one file matched:\n"
            f"{pattern}\n"
            f"in {folder}"
        )

    return files[0]


# ==================================================
# PC CORRELATIONS
# ==================================================

def pc_correlations(
    exact,
    randomized
):

    correlations = []

    for pc in range(K):

        corr = np.corrcoef(
            exact[:, pc],
            randomized[:, pc]
        )[0, 1]

        # PCA sign is arbitrary
        correlations.append(
            abs(corr)
        )

    return np.asarray(
        correlations
    )


# ==================================================
# LOAD EXACT PCA
# ==================================================

exact_ids, exact = read_pca(
    EXACT_FILE
)

print(
    "Exact PCA shape:",
    exact.shape
)


# ==================================================
# PARAMETER GRID
# ==================================================

median_correlation = np.full(
    (
        len(N_ITERS),
        len(N_OVERSAMPLES)
    ),
    np.nan
)

all_results = []


for i, n_iter in enumerate(N_ITERS):

    for j, n_over in enumerate(
        N_OVERSAMPLES
    ):

        run_folder = (
            RANDOM_ROOT
            / f"n{n_iter}_over{n_over}"
        )

        print(
            "\n"
            f"iter={n_iter}, "
            f"oversamples={n_over}"
        )

        seed_scores = []


        for seed in SEEDS:

            random_file = find_pca_file(
                run_folder,
                seed,
                n_iter,
                n_over
            )

            if random_file is None:

                print(
                    f"WARNING: "
                    f"missing seed {seed}"
                )

                continue


            random_ids, randomized = (
                read_pca(random_file)
            )


            # Ensure same cells / same order
            if not np.array_equal(
                exact_ids,
                random_ids
            ):

                raise ValueError(
                    f"Cell order differs "
                    f"for seed {seed}, "
                    f"iter={n_iter}, "
                    f"over={n_over}"
                )


            correlations = pc_correlations(
                exact,
                randomized
            )


            # --------------------------------------
            # One accuracy value for this seed:
            #
            # average absolute correlation across
            # PCs 1 ... 50
            # --------------------------------------

            mean_corr = np.mean(
                correlations
            )

            seed_scores.append(
                mean_corr
            )


            all_results.append(
                {
                    "n_iter": n_iter,
                    "n_oversamples": n_over,
                    "seed": seed,
                    "mean_abs_pc_correlation":
                        mean_corr,
                }
            )


        if len(seed_scores) == 0:
            continue


        # ------------------------------------------
        # Heatmap value =
        # median across 50 random seeds
        # ------------------------------------------

        median_correlation[i, j] = (
            np.median(seed_scores)
        )


        print(
            f"Runs found: "
            f"{len(seed_scores)}"
        )

        print(
            "Median mean |correlation|: "
            f"{median_correlation[i, j]:.4f}"
        )


# ==================================================
# SAVE RESULTS
# ==================================================

all_results_df = pd.DataFrame(
    all_results
)

all_results_df.to_csv(
    ALL_SEEDS_CSV,
    index=False
)


heatmap_df = pd.DataFrame(
    median_correlation,
    index=N_ITERS,
    columns=N_OVERSAMPLES
)

heatmap_df.index.name = "n_iter"
heatmap_df.columns.name = "n_oversamples"

heatmap_df.to_csv(
    OUTPUT_CSV
)


print(
    "\nMedian mean PC correlation:"
)

print(
    heatmap_df
)


# ==================================================
# HEATMAP
# ==================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


im = ax.imshow(
    median_correlation,
    cmap="viridis",
    aspect="auto",
    vmin=np.nanmin(median_correlation),
    vmax=1.0
)


# ==================================================
# AXES
# ==================================================

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
    "strong signal — individual PC recovery\n"
    "Median across 50 seeds, PCs 1–50"
)


# ==================================================
# NUMBERS INSIDE CELLS
# ==================================================

for i in range(len(N_ITERS)):

    for j in range(
        len(N_OVERSAMPLES)
    ):

        value = median_correlation[i, j]

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
# COLORBAR
# ==================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.03
)

cbar.set_label(
    "Median mean |PC correlation|\n"
    "(1.0 = exact individual PCs)"
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
    "Saved heatmap:",
    OUTPUT_CSV
)

print(
    "Saved all seed values:",
    ALL_SEEDS_CSV
)