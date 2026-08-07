import pandas as pd

summary = pd.read_csv(
    "test_results/pca_comparison/randomized_vs_arpack_summary.tsv",
    sep="\t",
)

b = summary["breakdown_pc"].dropna()

print("Breakdown PCs:")
print(b.to_list())

print("\nMean:", b.mean())
print("Median:", b.median())
print("Std dev:", b.std(ddof=1))
print("Min:", b.min())
print("Max:", b.max())
print("Range:", b.max() - b.min())