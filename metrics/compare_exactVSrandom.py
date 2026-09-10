from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

FULL_FILE = Path(
    "test_results/pca_full/p35_s42_broad_signal_full_pcas.tsv"
)

RANDOM_DIR = Path(
    "test_results/pca_randomized/35_broad_signal"
)

OUTPUT_FILE = Path(
    "figures/broad_signal.png"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# READ PCA
# --------------------------------------------------

def read_pca(path):

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=1,
        header=None
    )

    cell_ids = df.iloc[:, 0].astype(str).values
    scores = df.iloc[:, 1:].to_numpy(dtype=float)

    return cell_ids, scores


# exact / full PCA
exact_ids, exact = read_pca(FULL_FILE)


# --------------------------------------------------
# COMPARE ALL RANDOMIZED SEEDS
# --------------------------------------------------

results = []

random_files = sorted(
    RANDOM_DIR.glob("broad_signal_seed*_iter2_over10_pcas.tsv")
)

print("Randomized files found:", len(random_files))


for path in random_files:

    match = re.search(r"seed(\d+)", path.name)

    if match is None:
        continue

    seed = int(match.group(1))

    random_ids, randomized = read_pca(path)

    # sanity check
    if not np.array_equal(exact_ids, random_ids):
        raise ValueError(
            f"Cell order differs for seed {seed}"
        )

    correlations = []

    for pc in range(exact.shape[1]):

        corr = np.corrcoef(
            exact[:, pc],
            randomized[:, pc]
        )[0, 1]

        # PCA signs are arbitrary
        correlations.append(abs(corr))

    results.append(
        {
            "seed": seed,
            "correlations": correlations
        }
    )


# --------------------------------------------------
# PLOT ALL 50 SEEDS
# --------------------------------------------------

plt.figure(figsize=(11, 6))

pcs = np.arange(
    1,
    exact.shape[1] + 1
)

for result in results:

    plt.plot(
        pcs,
        result["correlations"],
        alpha=0.30,
        linewidth=1
    )


# old breakdown threshold
plt.axhline(
    0.8,
    linestyle="--",
    linewidth=1,
    label="Correlation threshold = 0.8"
)

plt.xlabel("Principal component")
plt.ylabel("|Correlation with full PCA|")
plt.title(
    "small gaps randomized PCA variability across 10 seeds"
)

plt.ylim(0, 1.02)
plt.xlim(1, 50)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300
)

plt.show()

print("Saved:", OUTPUT_FILE)