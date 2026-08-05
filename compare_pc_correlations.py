#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("test_results/pca_randomized")

SEED_1_FILE = (
    RESULTS_DIR
    / "pbmc3k_seed_1_pcas.tsv"
)

SEED_2_FILE = (
    RESULTS_DIR
    / "pbmc3k_seed_2_pcas.tsv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "seed_1_vs_seed_2_pc_correlations.tsv"
)

PLOT_FILE = (
    RESULTS_DIR
    / "seed_1_vs_seed_2_pc_correlations.png"
)


def load_scores(path):
    table = pd.read_csv(
        path,
        sep="\t",
    )

    if "cell_id" not in table.columns:
        raise ValueError(
            f"{path} does not contain a cell_id column"
        )

    table["cell_id"] = (
        table["cell_id"].astype(str)
    )

    return table.set_index("cell_id")


scores_1 = load_scores(SEED_1_FILE)
scores_2 = load_scores(SEED_2_FILE)


# Make sure both files contain the same cells
if set(scores_1.index) != set(scores_2.index):
    raise ValueError(
        "The two PCA outputs contain different cell IDs"
    )

# Reorder seed 2 to match seed 1 exactly
scores_2 = scores_2.loc[scores_1.index]


# Make sure both contain the same PC columns
if list(scores_1.columns) != list(scores_2.columns):
    raise ValueError(
        "The two PCA outputs contain different PC columns"
    )


pc_names = list(scores_1.columns)
absolute_correlations = []

for pc_name in pc_names:
    correlation = np.corrcoef(
        scores_1[pc_name].to_numpy(),
        scores_2[pc_name].to_numpy(),
    )[0, 1]

    absolute_correlations.append(
        abs(correlation)
    )


results = pd.DataFrame(
    {
        "pc": pc_names,
        "absolute_correlation": (
            absolute_correlations
        ),
    }
)

results.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
)


pc_numbers = np.arange(
    1,
    len(pc_names) + 1,
)

plt.figure(figsize=(9, 5))

plt.plot(
    pc_numbers,
    absolute_correlations,
    marker="o",
)

plt.xlabel("Principal component")
plt.ylabel("Absolute correlation")
plt.title(
    "PC correlations: randomized PCA seed 1 vs seed 2"
)
plt.ylim(0, 1.02)
plt.xticks(
    pc_numbers[::5]
)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=300,
)

plt.close()


print(results.to_string(index=False))

print("\nSaved table:")
print(OUTPUT_FILE)

print("\nSaved plot:")
print(PLOT_FILE)

print(
    "\nMean absolute correlation:",
    np.mean(absolute_correlations),
)

print(
    "Minimum absolute correlation:",
    np.min(absolute_correlations),
)