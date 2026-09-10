#!/usr/bin/env python3

from pathlib import Path
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


# --------------------------------------------------
# REPO IMPORT
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
from scipy.sparse import csr_matrix


def read_neighbors(path):
    with h5py.File(path, "r") as h5:

        cell_ids = h5["cell_ids"][:].astype(str)

        n = len(cell_ids)
        shape = (n, n)

        # distances
        distances = csr_matrix(
            (
                h5["data"][:],
                h5["indices"][:],
                h5["indptr"][:],
            ),
            shape=shape,
        )

        # connectivities
        g = h5["connectivities"]

        connectivities = csr_matrix(
            (
                g["data"][:],
                g["indices"][:],
                g["indptr"][:],
            ),
            shape=shape,
        )

    return distances, connectivities, cell_ids

# --------------------------------------------------
# BUILD UMAP ON EXACT GRAPH ONLY
# --------------------------------------------------

def compute_exact_umap(
    exact_neighbors_h5,
    random_seed=1,
):

    distances, connectivities, cell_ids = read_neighbors(
        exact_neighbors_h5
    )

    cell_ids = np.asarray(
        cell_ids,
        dtype=str,
    )

    adata = ad.AnnData(
        X=np.zeros(
            (len(cell_ids), 1)
        )
    )

    adata.obs_names = cell_ids

    adata.obsp["distances"] = distances
    adata.obsp["connectivities"] = connectivities

    # Approximate number of neighbors from stored graph
    n_neighbors = int(
        np.median(
            distances.getnnz(axis=1)
        )
    )

    adata.uns["neighbors"] = {
        "distances_key": "distances",
        "connectivities_key": "connectivities",
        "params": {
            "n_neighbors": n_neighbors,
            "method": "umap",
        },
    }

    sc.tl.umap(
        adata,
        random_state=random_seed,
    )

    return (
        adata.obsm["X_umap"].copy(),
        cell_ids,
    )





def load_cluster_labels(cluster_file, cell_ids):
    """
    Load cluster labels and align them to the fixed UMAP cell order.
    """

    df = pd.read_csv(
        cluster_file,
        sep="\t",
    )

    df = df.set_index("cell_id")

    if set(cell_ids) != set(df.index):
        raise ValueError(
            f"Cell IDs differ in {cluster_file}"
        )

    df = df.loc[cell_ids]

    return df["cluster"].astype(str).to_numpy()


def match_clusters_to_exact(
    exact_labels,
    randomized_labels,
):
    """
    Match every randomized Leiden cluster to the exact cluster
    with which it has the highest Jaccard overlap.

    Multiple randomized clusters may map to the same exact cluster.
    """

    exact_clusters = np.unique(exact_labels)
    random_clusters = np.unique(randomized_labels)

    mapping = {}

    for random_cluster in random_clusters:

        random_mask = (
            randomized_labels == random_cluster
        )

        best_exact = None
        best_jaccard = -1

        for exact_cluster in exact_clusters:

            exact_mask = (
                exact_labels == exact_cluster
            )

            intersection = np.sum(
                random_mask & exact_mask
            )

            union = np.sum(
                random_mask | exact_mask
            )

            jaccard = (
                intersection / union
                if union > 0
                else 0
            )

            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_exact = exact_cluster

        mapping[random_cluster] = best_exact

    return mapping


def remap_to_exact_labels(
    randomized_labels,
    mapping,
):
    """
    Replace each randomized cluster label by its matched exact cluster.
    """

    return np.array(
        [
            mapping[label]
            for label in randomized_labels
        ]
    )

# --------------------------------------------------
# PLOT SAME UMAP WITH DIFFERENT CLUSTERINGS
# --------------------------------------------------

def plot_cluster_comparison(
    umap,
    cell_ids,
    cluster_files,
    output_file,
):

    # Exact clustering determines the common colors
    exact_labels = load_cluster_labels(
        cluster_files["Exact PCA"],
        cell_ids,
    )

    exact_clusters = sorted(
        np.unique(exact_labels),
        key=lambda x: int(x),
    )

    cmap = plt.get_cmap(
        "turbo",
        len(exact_clusters),
    )

    color_map = {
        cluster: cmap(i)
        for i, cluster in enumerate(exact_clusters)
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12),
    )

    axes = axes.ravel()

    for ax, (title, cluster_file) in zip(
        axes,
        cluster_files.items(),
    ):

        labels = load_cluster_labels(
            cluster_file,
            cell_ids,
        )

        # Exact labels stay unchanged
        if title == "Exact PCA":
            display_labels = labels

        # Randomized clusters are matched
        # to closest exact clusters
        else:
            mapping = match_clusters_to_exact(
                exact_labels,
                labels,
            )

            display_labels = remap_to_exact_labels(
                labels,
                mapping,
            )

            print(f"\nMapping for {title}:")
            for random_cluster, exact_cluster in mapping.items():
                print(
                    f"  cluster {random_cluster}"
                    f" -> exact {exact_cluster}"
                )

        # Plot using SAME colors in every panel
        for cluster in exact_clusters:

            mask = display_labels == cluster

            ax.scatter(
                umap[mask, 0],
                umap[mask, 1],
                s=2,
                color=color_map[cluster],
                alpha=0.7,
            )

        ax.set_title(
            f"{title} "
            f"({len(np.unique(labels))} clusters)"
        )

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    plt.suptitle(
        "PBMC68k Leiden clustering on fixed exact-PCA UMAP\n"
        "Randomized clusters matched to exact clusters"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()

# --------------------------------------------------
# EXAMPLE: k = 36
# --------------------------------------------------

K = 36

EXACT_NEIGHBORS = Path(
    f"test_results/knn_pbmc68k/"
    f"pbmc68k_exact_k{K}_neighbors.h5"
)

CLUSTER_FILES = {

    "Exact PCA":
        Path(
            f"test_results/clusters_pbmc68k/"
            f"pbmc68k_exact_k{K}_clusters.tsv"
        ),

    "Best seed":
        Path(
            f"test_results/clusters_pbmc68k/"
            f"pbmc68k_best_k{K}_clusters.tsv"
        ),

    "Median seed":
        Path(
            f"test_results/clusters_pbmc68k/"
            f"pbmc68k_median_k{K}_clusters.tsv"
        ),

    "Worst seed":
        Path(
            f"test_results/clusters_pbmc68k/"
            f"pbmc68k_worst_k{K}_clusters.tsv"
        ),
}


OUTPUT_DIR = Path(
    "test_results/umap_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


umap, cell_ids = compute_exact_umap(
    EXACT_NEIGHBORS,
    random_seed=1,
)


plot_cluster_comparison(
    umap,
    cell_ids,
    CLUSTER_FILES,
    OUTPUT_DIR
    / f"pbmc68k_k{K}_cluster_comparison.png",
)


print(
    "Saved:",
    OUTPUT_DIR
    / f"pbmc68k_k{K}_cluster_comparison.png"
)