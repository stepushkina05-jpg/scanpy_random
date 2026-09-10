from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def compute_ari(cluster_file_a, cluster_file_b):
    """
    Compute Adjusted Rand Index (ARI) between two clustering results.

    Parameters
    ----------
    cluster_file_a : str or Path
        TSV file containing:
        cell_id    cluster

    cluster_file_b : str or Path
        TSV file containing:
        cell_id    cluster

    Returns
    -------
    float
        Adjusted Rand Index.

        ARI = 1:
            identical clustering

        ARI ~ 0:
            agreement approximately expected by chance
    """

    a = pd.read_csv(
        cluster_file_a,
        sep="\t",
    )

    b = pd.read_csv(
        cluster_file_b,
        sep="\t",
    )

    # --------------------------------------------------
    # CHECK CELL SETS
    # --------------------------------------------------

    if set(a["cell_id"]) != set(b["cell_id"]):
        raise ValueError(
            "The two clustering files contain different cells."
        )



    print("\nComparing:")
    print(cluster_file_a)
    print(cluster_file_b)

    print("Number of cells A:", len(a))
    print("Number of cells B:", len(b))

    # Duplicate cell IDs
    if a["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell IDs in file A")

    if b["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell IDs in file B")

    # Same cells
    if set(a["cell_id"]) != set(b["cell_id"]):
        raise ValueError(
            "The two clustering files contain different cells."
        )

    # Align

    a = a.set_index("cell_id")
    b = b.set_index("cell_id")

    b = b.loc[a.index]
    print("Number of clusters A:", a["cluster"].nunique())
    print("Number of clusters B:", b["cluster"].nunique())

    print("\nCluster sizes A:")
    print(a["cluster"].value_counts().sort_index())

    print("\nCluster sizes B:")
    print(b["cluster"].value_counts().sort_index())
    # --------------------------------------------------
    # ALIGN CELL ORDER
    # --------------------------------------------------

    



    

    # --------------------------------------------------
    # ARI
    # --------------------------------------------------

    ari = adjusted_rand_score(
        a["cluster"],
        b["cluster"],
    )

    return ari

EXACT_FILE = Path(
    "test_results/clusters_pbmc68k/"
    "pbmc68k_exact_k25_clusters.tsv"
)

BEST_FILE = Path(
    "test_results/clusters_pbmc68k/"
    "pbmc68k_best_k25_clusters.tsv"
)

MEDIAN_FILE = Path(
    "test_results/clusters_pbmc68k/"
    "pbmc68k_median_k25_clusters.tsv"
)

WORST_FILE = Path(
    "test_results/clusters_pbmc68k/"
    "pbmc68k_worst_k25_clusters.tsv"
)


ari_best = compute_ari(
    EXACT_FILE,
    BEST_FILE,
)

ari_median = compute_ari(
    EXACT_FILE,
    MEDIAN_FILE,
)

ari_worst = compute_ari(
    EXACT_FILE,
    WORST_FILE,
)


print("ARI vs exact clustering")
print("-----------------------")

print(
    f"Best seed:   {ari_best:.4f}"
)

print(
    f"Median seed: {ari_median:.4f}"
)

print(
    f"Worst seed:  {ari_worst:.4f}"
)
