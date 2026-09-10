import gzip
from pathlib import Path

import scanpy as sc


INPUT_DIR = Path(
    "/Users/dianastepushkina/Desktop/scanpy_data/simulated"
)

OUTPUT_ROOT = Path("test_input/simulated")

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# PROCESS EACH SIMULATED DATASET
# --------------------------------------------------

for input_file in sorted(INPUT_DIR.glob("*.h5ad")):

    dataset_name = input_file.stem.replace("_raw", "")

    print(f"\nProcessing: {dataset_name}")

    output_dir = OUTPUT_ROOT / dataset_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------
    # LOAD
    # ----------------------------------------------

    adata = sc.read_h5ad(input_file)

    # These exported .h5ad files have raw counts in X.
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # ----------------------------------------------
    # WRITE RAW DATA
    # ----------------------------------------------

    prepared_file = output_dir / "rawdata.h5ad"

    adata.write_h5ad(prepared_file)

    # ----------------------------------------------
    # CELL IDS
    # ----------------------------------------------

    with gzip.open(
        output_dir / "filtered_cellids.gz",
        "wt",
    ) as file:
        for cell_id in adata.obs_names:
            file.write(f"{cell_id}\n")

    # ----------------------------------------------
    # FEATURE IDS
    # ----------------------------------------------

    with gzip.open(
        output_dir / "filtered_featureids.gz",
        "wt",
    ) as file:
        for gene_id in adata.var_names:
            file.write(f"{gene_id}\n")

    print("Prepared raw data:", prepared_file)
    print("Cells:", adata.n_obs)
    print("Genes:", adata.n_vars)

    if "ground_truth" in adata.obs:
        print(
            "Ground-truth clusters:",
            adata.obs["ground_truth"].value_counts().to_dict()
        )