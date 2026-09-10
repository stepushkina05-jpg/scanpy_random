import scanpy as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from pathlib import Path

# --------------------------------------------------
# TRUE LABELS
# --------------------------------------------------

adata = sc.read_h5ad(
    "test_input/simulated/SimKumar8hard/rawdata.h5ad"
)

truth = pd.Series(
    adata.obs["ground_truth"].astype(str).values,
    index=adata.obs_names,
    name="ground_truth",
)

print("True clusters:")
print(truth.value_counts())


# --------------------------------------------------
# COMPARE CLUSTERING TO TRUE LABELS
# --------------------------------------------------

def compare_to_truth(cluster_file):

    clusters = pd.read_csv(
        cluster_file,
        sep="\t"
    )
    

    # first column = cell IDs
    # second column = Leiden cluster labels
    cell_ids = clusters.iloc[:, 0].astype(str)
    cluster_labels = clusters.iloc[:, 1].astype(str)

    predicted = pd.Series(
        cluster_labels.values,
        index=cell_ids,
        name="predicted"
    )

    common = truth.index.intersection(predicted.index)

    print("Cells compared:", len(common))

    ari = adjusted_rand_score(
        truth.loc[common],
        predicted.loc[common]
    )
    print("Number of true clusters:", truth.loc[common].nunique())
    print("Number of Leiden clusters:", predicted.loc[common].nunique())

    print("\nContingency table:")
    print( pd.crosstab( truth.loc[common], predicted.loc[common] ))
    
    return ari


files = {
    "best_k21":
        "test_results/cluster_8hard/best_21_1_clusters.tsv",

    "worst_k21":
        "test_results/cluster_8hard/worst_21_1_clusters.tsv",

    "exact_21": "test_results/cluster_8hard/full_k21_1_clusters.tsv"

}


for name, file in files.items():
    ari = compare_to_truth(file)
    print(f"{name}: ARI vs ground truth = {ari:.4f}")
# 1. Truth compared with itself MUST equal 1
print(
    "Truth vs truth:",
    adjusted_rand_score(truth, truth)
)