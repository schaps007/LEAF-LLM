"""Plot corrected OFAT summary data, with no hard-coded experimental values."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_factor_impact_poster(df, output_path):
    if "status" not in df:
        raise ValueError(
            "Legacy results lack validation status. "
            "Use a corrected run's ofat_results.csv"
        )

    df = df.loc[df.status.eq("ok")]

    baseline = df.loc[
        df.experiment_type.eq("baseline"),
        "energy_per_token",
    ]

    if len(baseline) != 1 or baseline.iloc[0] <= 0:
        raise ValueError(
            "One successful positive-energy baseline is required"
        )

    experiments = df.loc[
        df.experiment_type.eq("ofat")
    ].copy()

    if experiments.empty:
        raise ValueError("No successful comparisons to plot")

    experiments["change"] = 100 * (
        experiments.energy_per_token / baseline.iloc[0] - 1
    )
    experiments = experiments.sort_values("change")

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(experiments) * 0.5))
    )

    ax.barh(
        experiments.factor_value,
        experiments.change,
        color=[
            "#2ecc71" if v < 0 else "#e74c3c"
            for v in experiments.change
        ],
    )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(
        "Change in mean energy per output token relative to baseline (%)"
    )
    ax.set_title(
        "OFAT comparisons under the recorded execution conditions"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_csv",
        type=Path,
        help="Corrected run's ofat_results.csv",
    )
    args = parser.parse_args()

    output = args.results_csv.parent / "factor_impact_poster.png"

    plot_factor_impact_poster(
        pd.read_csv(args.results_csv),
        output,
    )

    print(f"Saved {output}")


if __name__ == "__main__":
    main()