from pathlib import Path
import re
import numpy as np
import pandas as pd

RANDOM_ROOT = Path("test_results/pca_randomized")
OUT = Path("scanpy_pc_correlations.csv")
N_COMPONENTS = 50

DATASETS = ["31_narrow_signal", "35_broad_signal", "38_strong_signal"]

EXACT_FILES = {
    "31_narrow_signal": Path("test_results/pca_full/p31_s42_narrow_signal_full_pcas.tsv"),
    "35_broad_signal": Path("test_results/pca_full/p35_s42_broad_signal_full_pcas.tsv"),
    "38_strong_signal": Path("test_results/pca_full/p38_s42_strong_signal_full_pcas.tsv"),
}


def read_pcas(path):
    return pd.read_csv(path, sep="\t", index_col=0).iloc[:, :N_COMPONENTS]


def seed_from_name(path):
    m = re.search(r"seed[_-]?(\d+)", path.stem, re.I)
    if not m:
        raise ValueError(f"Cannot find seed in {path.name}")
    return int(m.group(1))


rows = []

for dataset in DATASETS:
    dataset_dir = RANDOM_ROOT / dataset
    exact = read_pcas(EXACT_FILES[dataset])

    print(f"\n{dataset}")

    for param_dir in sorted(dataset_dir.iterdir()):
        m = re.fullmatch(r"n(\d+)_over(\d+)", param_dir.name)
        if not param_dir.is_dir() or not m:
            continue

        n_iter, n_over = int(m.group(1)), int(m.group(2))

        # Only the actual experiment grid
        if n_iter not in {2, 3, 4, 5} or n_over not in {10, 20, 30, 40}:
            continue

        files = sorted(param_dir.glob("*_pcas.tsv"), key=seed_from_name)

        if len(files) != 50:
            print(f"WARNING {dataset}/{param_dir.name}: found {len(files)} files")

        for pca_file in files:
            seed = seed_from_name(pca_file)
            rand = read_pcas(pca_file).reindex(exact.index)

            if rand.isna().any().any():
                raise RuntimeError(f"Cell IDs do not align: {pca_file}")

            row = {
                "dataset": dataset,
                "n_iter": n_iter,
                "n_oversamples": n_over,
                "seed": seed,
                "n_components": N_COMPONENTS,
            }

            for k in range(N_COMPONENTS):
                corr = np.corrcoef(exact.iloc[:, k], rand.iloc[:, k])[0, 1]
                row[f"PC{k+1}_corr"] = abs(corr)

            rows.append(row)

        print(f"  finished n_iter={n_iter}, oversamples={n_over}")


df = pd.DataFrame(rows).sort_values(
    ["dataset", "n_iter", "n_oversamples", "seed"]
).reset_index(drop=True)

df.to_csv(OUT, index=False)

print("\n========================================")
print(f"Rows: {len(df)} (expected 2400)")
print(f"Columns: {len(df.columns)} (expected 55)")
print(f"Wrote: {OUT}")
print("========================================")
print(df.head().to_string(index=False))