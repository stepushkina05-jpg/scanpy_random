from pathlib import Path
import json, re
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

RANDOM_ROOT = Path("test_results/pca_randomized")
EXACT_ROOT = Path("test_results/pca_full")
OUT = Path("scanpy_master_runs.csv")
N_COMPONENTS = 50

INPUTS = {
    "35_broad_signal": Path("test_results/feat_select/p35_s42_broad_signal_normalized_selected.h5"),
    "31_narrow_signal": Path("test_results/feat_select/p31_s42_narrow_signal_normalized_selected.h5"),
    "38_strong_signal": Path("test_results/feat_select/p38_s42_strong_signal_normalized_selected.h5"),
}
EXACT_FILES = {
    "31_narrow_signal": Path("test_results/pca_full/p31_s42_narrow_signal_full_pcas.tsv"),
    "35_broad_signal": Path("test_results/pca_full/p35_s42_broad_signal_full_pcas.tsv"),
    "38_strong_signal": Path("test_results/pca_full/p38_s42_strong_signal_full_pcas.tsv"),
}

def read_tsv(path):
    return pd.read_csv(path, sep="\t", index_col=0)


def load_X(path):
    with h5py.File(path, "r") as h5:
        g = h5["matrix"]
        X = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                          shape=tuple(g["shape"][:])).T.toarray()
        cells = g["barcodes"][:].astype(str)
        genes = g["genes"][:].astype(str)
    return X, cells, genes


def seed_from_name(path):
    m = re.search(r"seed[_-]?(\d+)", path.stem, re.I)
    if not m:
        raise ValueError(f"Cannot find seed in {path.name}")
    return int(m.group(1))


def spectral_norm(A):
    return float(svds(A, k=1, which="LM", return_singular_vectors=False)[0])


def pc_correlation(exact, rand):
    corrs = [abs(np.corrcoef(exact[:, k], rand[:, k])[0, 1]) for k in range(N_COMPONENTS)]
    return float(np.mean(corrs))


def runtimes_from_log(path):
    runtimes, start = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("event") != "pca":
                continue
            if e.get("phase") == "start":
                start = pd.to_datetime(e["ts"], utc=True)
            elif e.get("phase") == "end":
                if start is None:
                    raise RuntimeError(f"PCA end without start in {path}")
                end = pd.to_datetime(e["ts"], utc=True)
                runtimes.append((end - start).total_seconds())
                start = None
    return {seed: t for seed, t in enumerate(runtimes, start=1)}


rows = []

DATASETS = ["31_narrow_signal", "35_broad_signal", "38_strong_signal"]

for dataset in DATASETS:
    dataset_dir = RANDOM_ROOT / dataset

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {dataset_dir}")
    print(f"\n{dataset}")

    # ---------- exact PCA ----------
    exact_pca_file = EXACT_FILES[dataset]

    if not exact_pca_file.exists():
        raise FileNotFoundError(f"Missing full PCA file: {exact_pca_file}")
    exact_load_file = Path(str(exact_pca_file).replace("_pcas.tsv", "_loadings.tsv"))

    exact_scores_df = read_tsv(exact_pca_file)
    exact_load_df = read_tsv(exact_load_file)

    # ---------- original normalized/selected matrix ----------
    X, cells, genes = load_X(INPUTS[dataset])

    cell_pos = pd.Index(cells).get_indexer(exact_scores_df.index)
    gene_pos = pd.Index(genes).get_indexer(exact_load_df.index)
    if (cell_pos < 0).any() or (gene_pos < 0).any():
        raise RuntimeError(f"Cell/gene IDs do not align for {dataset}")

    X = X[cell_pos][:, gene_pos]
    Xc = X - X.mean(axis=0)

    exact_scores = exact_scores_df.iloc[:, :N_COMPONENTS].to_numpy()
    exact_loadings = exact_load_df.iloc[:, :N_COMPONENTS].to_numpy()

    # exact rank-50 residual only needs to be computed once per dataset
    exact_residual_norm = spectral_norm(Xc - exact_scores @ exact_loadings.T)
    print(f"  exact residual norm = {exact_residual_norm:.6f}")

    # ---------- randomized parameter combinations ----------
    for param_dir in sorted(dataset_dir.iterdir()):
        m = re.fullmatch(r"n(\d+)_over(\d+)", param_dir.name)
        if not param_dir.is_dir() or not m:
            continue

        n_iter, n_over = int(m.group(1)), int(m.group(2))
        runtime = runtimes_from_log(param_dir / "obkit-events.jsonl")
        random_files = sorted(param_dir.glob("*_pcas.tsv"), key=seed_from_name)

        if len(random_files) != 50:
            print(f"  WARNING {param_dir.name}: {len(random_files)} PCA files")
        if len(runtime) != 50:
            print(f"  WARNING {param_dir.name}: {len(runtime)} runtimes")

        for pca_file in random_files:
            seed = seed_from_name(pca_file)
            load_file = Path(str(pca_file).replace("_pcas.tsv", "_loadings.tsv"))

            rand_scores_df = read_tsv(pca_file).reindex(exact_scores_df.index)
            rand_load_df = read_tsv(load_file).reindex(exact_load_df.index)

            rand_scores = rand_scores_df.iloc[:, :N_COMPONENTS].to_numpy()
            rand_loadings = rand_load_df.iloc[:, :N_COMPONENTS].to_numpy()

            mean_corr = pc_correlation(exact_scores, rand_scores)

            rand_residual_norm = spectral_norm(Xc - rand_scores @ rand_loadings.T)
            approximation_quality = exact_residual_norm / rand_residual_norm

            rows.append({
                "implementation": "scanpy",
                "dataset": dataset,
                "n_iter": n_iter,
                "n_oversamples": n_over,
                "seed": seed,
                "n_components": N_COMPONENTS,
                "approximation_quality": approximation_quality,
                "mean_abs_pc_corr": mean_corr,
                "runtime_s": runtime.get(seed, np.nan),
            })

        print(f"  finished n_iter={n_iter}, oversamples={n_over}")


master = pd.DataFrame(rows).sort_values(
    ["dataset", "n_iter", "n_oversamples", "seed"]
).reset_index(drop=True)

master.to_csv(OUT, index=False)

print("\n========================================")
print(f"Rows: {len(master)} (expected 2400)")
print(f"Missing runtimes: {master['runtime_s'].isna().sum()}")
print(f"Parameter combinations: {master.groupby(['dataset','n_iter','n_oversamples']).ngroups}")
print(f"Wrote: {OUT}")
print("========================================")

print(master.head(10).to_string(index=False))