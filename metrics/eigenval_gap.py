from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd


def compute_relative_spectral_gaps(pca_file):
    """
    Compute relative spectral gaps from exact PCA scores.

    Relative gap after PC k:
        (lambda_k - lambda_{k+1}) / lambda_k

    Parameters
    ----------
    pca_file : str or Path
        TSV containing PCA scores.
        First column must contain cell IDs.

    Returns
    -------
    pd.DataFrame
        Columns:
        - k
        - eigenvalue_k
        - eigenvalue_next
        - relative_gap
    """

    # Load PCA scores
    scores = pd.read_csv( pca_file, sep="\t", index_col=0, )

    scores = scores.apply( pd.to_numeric, errors="raise",)

    # Variance of each exact PC = corresponding eigenvalue
    eigenvalues = ( scores.var( axis=0,ddof=1,) .to_numpy(dtype=np.float64) )

    # Relative spectral gap
    relative_gaps = (eigenvalues[:-1] - eigenvalues[1:] ) / eigenvalues[:-1]

    gap_table = pd.DataFrame(
        {
            "k": np.arange(1, len(eigenvalues),),
            "eigenvalue_k": eigenvalues[:-1],
            "eigenvalue_next": eigenvalues[1:],
            "relative_gap": relative_gaps,
        }
    )

    return gap_table

EXACT_FILE = Path( "test_results/pca_full/p35_s42_broad_signal_full_pcas.tsv")
gaps = compute_relative_spectral_gaps(EXACT_FILE)
print(gaps)

def plot_spectral_gaps(gaps, output_file):

    smallest = gaps.nsmallest( 5,"relative_gap",)

    largest = gaps.nlargest( 5, "relative_gap",)

    plt.figure(figsize=(10, 6))

    # all gaps
    plt.plot(
        gaps["k"],
        gaps["relative_gap"],
        marker="o",
        markersize=3,
        label="Relative spectral gap",
    )

    # smallest gaps
    plt.scatter(
        smallest["k"],
        smallest["relative_gap"],
        s=80,
        label="5 smallest gaps",
    )

    # largest gaps
    plt.scatter(
        largest["k"],
        largest["relative_gap"],
        s=80,
        label="5 largest gaps",
    )

    # label selected k values
    for _, row in smallest.iterrows():
        plt.annotate(
            f"k={int(row['k'])}",(row["k"], row["relative_gap"]),xytext=(5, 5), textcoords="offset points",)

    for _, row in largest.iterrows():
        plt.annotate(f"k={int(row['k'])}",(row["k"], row["relative_gap"]),xytext=(5, 5),textcoords="offset points",)

    plt.xlabel("Number of retained PCs (k)")
    plt.ylabel("Relative spectral gap")
    plt.title("large relative spectral gaps")
    plt.legend()
    plt.tight_layout()
    plt.savefig( output_file,dpi=300,)
    plt.show()

OUTPUT_FILE = Path("test_results/eigenvalue_gaps/p35_broad_signal.png")
OUTPUT_FILE.parent.mkdir( parents=True,exist_ok=True,)

print("\n5 smallest gaps:")
print(gaps.nsmallest(5,"relative_gap",))
print("\n5 largest gaps:")
print(gaps.nlargest(5,"relative_gap",))
plot_spectral_gaps(gaps,OUTPUT_FILE,)