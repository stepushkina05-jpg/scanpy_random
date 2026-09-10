import gzip
from pathlib import Path

import scanpy as sc


INPUT_PATH = Path(
    "/Users/dianastepushkina/Desktop/scanpy_data/pbmc68k"
)

OUTPUT_DIR = Path("test_input")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# LOAD INPUT
# --------------------------------------------------

if INPUT_PATH.is_file() and INPUT_PATH.suffix == ".h5ad":
    adata = sc.read_h5ad(INPUT_PATH)

elif INPUT_PATH.is_dir():
    adata = sc.read_10x_mtx(
        INPUT_PATH,
        var_names="gene_symbols",
        make_unique=True,
        cache=False,
    )

else:
    raise ValueError(
        f"Unsupported input: {INPUT_PATH}"
    )


# The repo expects raw counts here.
# Only do this if adata.X really contains raw counts.

if "counts" not in adata.layers:
    adata.layers["counts"] = adata.X.copy()


prepared_file = OUTPUT_DIR / "rawdata.h5ad"
adata.write_h5ad(prepared_file)


with gzip.open(
    OUTPUT_DIR / "filtered_cellids.gz",
    "wt",
) as file:
    for cell_id in adata.obs_names:
        file.write(f"{cell_id}\n")


with gzip.open(
    OUTPUT_DIR / "filtered_featureids.gz",
    "wt",
) as file:
    for gene_id in adata.var_names:
        file.write(f"{gene_id}\n")


print("Prepared raw data:", prepared_file)
print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)