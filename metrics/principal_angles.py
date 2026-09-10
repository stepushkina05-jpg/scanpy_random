#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles


# --------------------------------------------------
# PATHS
# --------------------------------------------------

EXACT_FILE = Path(
    "test_results/pca_full/8hard_full_loadings.tsv"
)

SEED_DIR = Path(
    "test_results/pca_randomized/8hard"
)

OUTPUT_FILE = Path(
    "test_results/8hard_subspace_errors_selected_k.csv"
)


# --------------------------------------------------
# SELECTED k VALUES
# Replace these with your 5 small-gap + 5 large-gap k's
# --------------------------------------------------

SELECTED_K = [ 2, 5, 21, 27, 31, 41, 45]
    
SEEDS = range(1, 51)


# --------------------------------------------------
# LOAD LOADINGS
# --------------------------------------------------

def load_loadings(path):
    df = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
    )

    return df.astype(np.float64)


# --------------------------------------------------
# SUBSPACE ERROR
# --------------------------------------------------

def compute_subspace_error(
    exact_loadings,
    randomized_loadings,
    k,
):
    """
    Compare the first k-dimensional PCA subspaces.

    Returns
    -------
    subspace_error :
        sqrt(mean(sin(theta_i)^2))

        0 = identical subspaces
        1 = maximally different

    max_angle_deg :
        Largest principal angle in degrees.
    """

    U_exact = exact_loadings.iloc[:, :k].to_numpy()

    U_random = randomized_loadings.iloc[:, :k].to_numpy()

    angles = subspace_angles(
        U_exact,
        U_random,
    )

    subspace_error = np.sqrt(
        np.mean(
            np.sin(angles) ** 2
        )
    )

    max_angle_deg = np.degrees(
        np.max(angles)
    )

    return (
        subspace_error,
        max_angle_deg,
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

exact = load_loadings(
    EXACT_FILE
)

results = []


for seed in SEEDS:

    seed_file = (
        SEED_DIR
        / f"8hard_seed_{seed}_loadings.tsv"
    )

    if not seed_file.exists():
        print(
            f"Missing seed {seed}: {seed_file}"
        )
        continue

    randomized = load_loadings(
        seed_file
    )

    # Make sure genes are identical and in same order
    randomized = randomized.loc[
        exact.index
    ]

    for k in SELECTED_K:

        error, max_angle = compute_subspace_error(
            exact,
            randomized,
            k,
        )

        results.append(
            {
                "seed": seed,
                "k": k,
                "subspace_error": error,
                "max_angle_deg": max_angle,
            }
        )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

results = pd.DataFrame(
    results
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nSubspace error summary")
print("----------------------")

for k in SELECTED_K:

    subset = results[
        results["k"] == k
    ]

    best = subset.loc[
        subset["subspace_error"].idxmin()
    ]

    worst = subset.loc[
        subset["subspace_error"].idxmax()
    ]

    median_value = subset[
        "subspace_error"
    ].median()

    median = subset.iloc[
        (
            subset["subspace_error"]
            - median_value
        )
        .abs()
        .argsort()[:1]
    ].iloc[0]

    print(f"\nk = {k}")

    print(
        f"  BEST:   seed {int(best['seed'])}, "
        f"error = {best['subspace_error']:.4f}"
    )

    print(
        f"  MEDIAN: seed {int(median['seed'])}, "
        f"error = {median['subspace_error']:.4f}"
    )

    print(
        f"  WORST:  seed {int(worst['seed'])}, "
        f"error = {worst['subspace_error']:.4f}"
    )