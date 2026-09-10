import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# PATH TO EXACT PCA
# --------------------------------------------------

PCA_FILE = "test_results/pca_full/p35_s42_broad_signal_full_pcas.tsv"

DATASET_NAME = "p35_s42_broad_signal"


# --------------------------------------------------
# LOAD EXACT PCA SCORES
# --------------------------------------------------

df = pd.read_csv(PCA_FILE, sep="\t")

# first column is cell_id
scores = df.iloc[:, 1:].to_numpy(dtype=float)


# --------------------------------------------------
# COMPUTE PCA EIGENVALUES
# --------------------------------------------------

# Variance of each exact PCA score = corresponding eigenvalue
eigenvalues = np.var(scores, axis=0, ddof=1)

pcs = np.arange(1, len(eigenvalues) + 1)


# --------------------------------------------------
# QUANTIFY SPECTRAL DECAY
# --------------------------------------------------

# Normalize so lambda_1 = 1
normalized_eigenvalues = eigenvalues / eigenvalues[0]


# 1. Overall exponential decay slope: PCs 1-50
overall_slope, overall_intercept = np.polyfit(
    pcs,
    np.log(eigenvalues),
    1
)


# 2. Tail decay slope: PCs 10-50
# Useful because the first few dominant PCs can otherwise control the slope
tail_start = 10

tail_mask = pcs >= tail_start

tail_slope, tail_intercept = np.polyfit(
    pcs[tail_mask],
    np.log(eigenvalues[tail_mask]),
    1
)


# Convert slopes into approximate % decrease in eigenvalue per PC
overall_decay_per_pc = 1 - np.exp(overall_slope)
tail_decay_per_pc = 1 - np.exp(tail_slope)


# 3. How much eigenvalue remains at PC50 relative to PC1?
end_ratio = eigenvalues[-1] / eigenvalues[0]


# 4. How many times larger is lambda_1 than lambda_50?
drop_factor = eigenvalues[0] / eigenvalues[-1]


print("\n--- Spectral decay ---")
print(f"Overall log-slope:       {overall_slope:.4f}")
print(f"Tail log-slope (10-50): {tail_slope:.4f}")

print(
    f"Overall decay per PC:   "
    f"{overall_decay_per_pc * 100:.2f}%"
)

print(
    f"Tail decay per PC:      "
    f"{tail_decay_per_pc * 100:.2f}%"
)

print(f"lambda_50 / lambda_1:   {end_ratio:.4f}")
print(f"lambda_1 / lambda_50:   {drop_factor:.2f}x")


# --------------------------------------------------
# PLOT NORMALIZED SPECTRUM
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    pcs,
    normalized_eigenvalues,
    marker="o",
    markersize=4
)

plt.yscale("log")

plt.xlabel("Principal component (k)")
plt.ylabel(r"Relative eigenvalue $\lambda_k / \lambda_1$")
plt.title(f"{DATASET_NAME} PCA eigenvalue spectrum")

plt.tight_layout()


# --------------------------------------------------
# SAVE FIGURE
# --------------------------------------------------

OUTPUT_DIR = Path("spectrum")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_DIR / f"{DATASET_NAME}_spectrum.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()