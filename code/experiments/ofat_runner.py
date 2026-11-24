"""OFAT runner preserving every repetition, failures, and reproducibility metadata."""
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata
import json
import platform
import random
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factors.search_space import BASELINE_CONFIG, get_ofat_experiments
from experiments.run_experiment import run_single_experiment
from workloads.loader import load_workload
from utils.seed import set_seed


def load_config():
    with (ROOT / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)

    if config.get("repetitions", 0) < 2:
        raise ValueError(
            "At least two independent repetitions are required"
        )

    return config


def successful_results(df):
    return df.loc[df["status"].eq("ok")].copy()


def summarize_results(raw):
    valid = successful_results(raw)

    if valid.empty:
        return valid

    keys = [
        "configuration_id",
        "experiment_type",
        "factor",
        "factor_value",
        "model",
        "quantization",
        "batch_size",
        "workload",
        "hardware",
        "measurement_method",
    ]

    excluded = {"repetition", "seed", "execution_order"}

    numeric = [
        c
        for c in valid.select_dtypes(include="number").columns
        if c not in keys and c not in excluded
    ]

    summary = (
        valid.groupby(keys, dropna=False)[numeric]
        .mean()
        .reset_index()
    )

    counts = raw.groupby("configuration_id").size()
    successes = valid.groupby("configuration_id").size()

    summary["n_successful"] = summary.configuration_id.map(successes)
    summary["n_attempted"] = summary.configuration_id.map(counts)
    summary["status"] = "ok"

    return summary


def run_ofat_experiments(config, output_dir=None):
    baseline = dict(
        BASELINE_CONFIG,
        configuration_id="baseline",
        experiment_type="baseline",
        factor="baseline",
        factor_value="Baseline",
    )

    experiments = [baseline]

    for i, exp in enumerate(get_ofat_experiments(), 1):
        experiments.append(
            dict(
                exp,
                configuration_id=f"ofat_{i:03d}",
                experiment_type="ofat",
                factor_value=exp["description"],
            )
        )

    # Each repetition is a randomized block: baseline drift is not tied
    # to one start time.
    rng = random.Random(config["seed"])
    records = []

    for rep in range(1, config["repetitions"] + 1):
        block = experiments.copy()
        rng.shuffle(block)

        for exp in block:
            seed = config["seed"] + rep - 1
            set_seed(seed)
            prompts = load_workload(exp["workload"])

            if exp["batch_size"] > len(prompts):
                result = {
                    "status": "failed",
                    "error": "Batch size exceeds workload size; add prompts",
                }
            else:
                result = run_single_experiment(
                    config,
                    exp["model"],
                    prompts,
                    exp["batch_size"],
                    quantization=exp["quantization"],
                    hardware=exp["hardware"],
                    measurement_method=exp["measurement_method"],
                )

            result.update(
                {
                    k: v
                    for k, v in exp.items()
                    if k != "description"
                }
            )

            result.update(
                repetition=rep,
                seed=seed,
                execution_order=len(records) + 1,
            )
            records.append(result)

            print(
                f"{exp['configuration_id']} rep {rep}: "
                f"{result['status']} {result.get('error', '')}"
            )

            if output_dir is not None:
                save_results(
                    pd.DataFrame(records),
                    "raw_results.csv",
                    output_dir,
                )

    return pd.DataFrame(records)


def calculate_confidence_intervals(
    raw,
    metric="energy_per_token",
):
    from scipy import stats

    rows = []
    valid = successful_results(raw)

    if metric not in valid:
        return pd.DataFrame()

    for config_id, group in valid.groupby("configuration_id"):
        values = group[metric].dropna().to_numpy()
        n = len(values)
        mean = float(np.mean(values))

        std = (
            float(np.std(values, ddof=1))
            if n >= 2
            else None
        )

        half = (
            float(
                stats.t.ppf(0.975, n - 1)
                * std
                / np.sqrt(n)
            )
            if n >= 2
            else None
        )

        rows.append(
            dict(
                configuration_id=config_id,
                metric=metric,
                mean=mean,
                std=std,
                n=n,
                ci_lower=(
                    mean - half
                    if half is not None
                    else None
                ),
                ci_upper=(
                    mean + half
                    if half is not None
                    else None
                ),
            )
        )

    return pd.DataFrame(rows)


def statistical_analysis(raw):
    """Paired block tests; Holm correction across all completed comparisons."""
    from scipy import stats

    valid = successful_results(raw)

    if "energy_per_token" not in valid:
        return pd.DataFrame()

    baseline = valid.loc[
        valid.experiment_type.eq("baseline"),
        ["repetition", "energy_per_token"],
    ]

    rows = []

    experiments = valid.loc[
        valid.experiment_type.eq("ofat")
    ]

    for config_id, group in experiments.groupby("configuration_id"):
        paired = baseline.merge(
            group[["repetition", "energy_per_token"]],
            on="repetition",
            suffixes=("_base", "_exp"),
        )

        if len(paired) < 3:
            continue

        differences = (
            paired.energy_per_token_exp
            - paired.energy_per_token_base
        )

        if np.all(differences == 0):
            p = 1.0
        else:
            p = float(
                stats.ttest_rel(
                    paired.energy_per_token_exp,
                    paired.energy_per_token_base,
                ).pvalue
            )

        if not np.isfinite(p):
            continue

        rows.append(
            dict(
                configuration_id=config_id,
                n_pairs=len(paired),
                p_value=p,
                mean_difference=float(differences.mean()),
            )
        )

    rows.sort(key=lambda row: row["p_value"])

    previous = 0.0

    for rank, row in enumerate(rows):
        previous = max(
            previous,
            min(
                1.0,
                (len(rows) - rank) * row["p_value"],
            ),
        )

        row.update(
            p_value_holm=previous,
            significant=previous < 0.05,
        )

    return pd.DataFrame(rows)


def analyze_ofat_results(summary):
    if (
        summary.empty
        or not summary.experiment_type.eq("baseline").any()
    ):
        return pd.DataFrame()

    baseline = summary.loc[
        summary.experiment_type.eq("baseline")
    ].iloc[0]

    rows = []

    for _, exp in summary.loc[
        summary.experiment_type.eq("ofat")
    ].iterrows():
        row = dict(
            configuration_id=exp.configuration_id,
            factor=exp.factor,
            value=exp.factor_value,
        )

        for prefix, metric in (
            ("energy", "energy_per_token"),
            ("carbon", "carbon_per_query"),
        ):
            row[metric] = exp[metric]
            row[f"{prefix}_change"] = (
                exp[metric] - baseline[metric]
            )
            row[f"{prefix}_pct"] = (
                100
                * row[f"{prefix}_change"]
                / baseline[metric]
                if baseline[metric]
                else None
            )

        rows.append(row)

    return pd.DataFrame(rows)


def save_results(
    df,
    filename="ofat_results.csv",
    output_dir=None,
):
    folder = (
        Path(output_dir)
        if output_dir
        else ROOT / "results"
    )

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    df.to_csv(path, index=False)

    return path


def main():
    config = load_config()

    # Preserve the original thesis data; corrected runs always get
    # a separate directory.
    output = (
        ROOT
        / "results"
        / datetime.now(timezone.utc).strftime(
            "run_%Y%m%dT%H%M%S_%fZ"
        )
    )
    output.mkdir(parents=True)

    packages = {}

    for name in (
        "torch",
        "transformers",
        "codecarbon",
        "numpy",
        "pandas",
        "scipy",
        "bitsandbytes",
        "accelerate",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    manifest = dict(
        config=config,
        python=sys.version,
        platform=platform.platform(),
        processor=platform.processor(),
        packages=packages,
        baseline=BASELINE_CONFIG,
        energy_boundary=(
            "post-warmup inference including transfers; "
            "excludes load/tokenization"
        ),
        workloads={
            name: load_workload(name)
            for name in ("short", "medium", "long")
        },
    )

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    raw = run_ofat_experiments(config, output)
    summary = summarize_results(raw)

    save_results(
        summary,
        "ofat_results.csv",
        output,
    )

    save_results(
        analyze_ofat_results(summary),
        "impact_analysis.csv",
        output,
    )

    if not summary.empty:
        for metric in (
            "energy_per_token",
            "carbon_per_query",
        ):
            save_results(
                calculate_confidence_intervals(raw, metric),
                f"{metric}_confidence_intervals.csv",
                output,
            )

        save_results(
            statistical_analysis(raw),
            "statistical_analysis.csv",
            output,
        )

    print(f"Saved results to {output}")

    if successful_results(raw).empty:
        raise RuntimeError(
            "All experiments failed; inspect raw_results.csv"
        )


if __name__ == "__main__":
    main()