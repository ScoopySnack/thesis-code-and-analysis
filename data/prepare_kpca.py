import os
import sys
import json
import math
import pandas as pd
import numpy as np

# This script prepares a numeric feature matrix normalized for KPCA.
# - Loads a CSV (default: data/alkanes_core_with_smiles_final_with_graph.csv)
# - Selects numeric columns automatically
# - Imputes missing numeric values with the column median
# - Standardizes each numeric column to zero mean and unit variance (population std)
# - Writes a new CSV that PRESERVES original columns and adds normalized numeric columns with '__norm' suffix
# - Saves a small JSON with the preprocessing parameters for reproducibility


def _default_paths(in_csv: str | None, out_csv: str | None):
    # Resolve defaults relative to this script's directory to be robust to CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(script_dir, "alkanes_core_with_smiles_final_with_graph.csv")
    if in_csv is None:
        in_csv = default_in
    if out_csv is None:
        base, ext = os.path.splitext(in_csv)
        out_csv = base + "_normalized.csv"
    return in_csv, out_csv


def _is_finite_number(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def normalize_for_kpca(in_csv: str | None = None, out_csv: str | None = None, params_json: str | None = None):
    in_csv, out_csv = _default_paths(in_csv, out_csv)
    if params_json is None:
        base, _ = os.path.splitext(out_csv)
        params_json = base + "__scaler_params.json"

    print(f"Reading: {in_csv}")
    df = pd.read_csv(in_csv)

    # Identify numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        raise RuntimeError("No numeric columns found to normalize.")

    # Prepare arrays for means, stds, medians
    medians = {}
    means = {}
    stds = {}

    # Work on a copy to avoid modifying original df until the end
    out_df = df.copy()

    normalized_cols = []

    for col in num_cols:
        col_series = out_df[col]
        # Compute median ignoring non-finite values (coerce first)
        # Pandas numeric dtype should already be numeric; still handle NaNs
        median_val = float(col_series.median()) if not col_series.dropna().empty else 0.0
        # Impute NaNs (non-finite values should already be NaN in numeric dtype)
        col_imputed = col_series.fillna(median_val)

        # Compute mean and std with population denominator (ddof=0), matching sklearn's StandardScaler
        mean_val = float(col_imputed.mean()) if len(col_imputed) else 0.0
        std_val = float(col_imputed.std(ddof=0)) if len(col_imputed) else 1.0
        if std_val == 0.0 or not math.isfinite(std_val):
            std_val = 1.0  # avoid division by zero; results will be zeros

        # Save params
        medians[col] = median_val
        means[col] = mean_val
        stds[col] = std_val

        # Standardize into a new column with suffix
        norm_col = f"{col}__norm"
        out_df[norm_col] = (col_imputed - mean_val) / std_val
        normalized_cols.append(norm_col)

    # Save normalized CSV
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(out_csv, index=False)
    print(f"Saved normalized data for KPCA: {out_csv}")

    # Save parameters JSON
    params = {
        "input_csv": in_csv,
        "output_csv": out_csv,
        "numeric_columns": normalized_cols,
        "impute_median": medians,
        "standardize_mean": means,
        "standardize_std": stds,
        "notes": "Original columns were preserved. Normalized counterparts were added with '__norm' suffix. Numeric columns were imputed with column median and standardized to zero mean and unit variance (population std).",
    }

    with open(params_json, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Saved preprocessing parameters: {params_json}")


def main():
    # CLI usage: python data/prepare_kpca.py [input_csv] [output_csv]
    in_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    normalize_for_kpca(in_arg, out_arg)


if __name__ == "__main__":
    main()
