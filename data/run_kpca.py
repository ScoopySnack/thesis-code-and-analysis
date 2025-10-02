import os
import sys
import json
import math
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances


# This script performs Kernel PCA on the normalized dataset using an RBF kernel
# and computes the "correct" gamma via the median heuristic:
#   gamma = 1 / (2 * median(pairwise_distances(X))^2)
# It appends KPCA components to the original rows and saves metadata including
# the gamma value used.


def _default_paths(in_csv: Optional[str], out_csv: Optional[str]):
    # Resolve defaults relative to this script's directory to be robust to CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(script_dir, "numeric_only_normalized.csv")
    if in_csv is None:
        in_csv = default_in
    if out_csv is None:
        base, ext = os.path.splitext(in_csv)
        out_csv = base + "__kpca.csv"
    return in_csv, out_csv


def _scaler_params_path(in_csv: str, params_json: Optional[str]):
    if params_json is not None:
        return params_json
    base, _ = os.path.splitext(in_csv)
    # If input already ends with _normalized.csv this will align with prepare_kpca.py
    return base + "__scaler_params.json"


def _load_numeric_columns(params_path: str, df: pd.DataFrame) -> list[str]:
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        cols = params.get("numeric_columns")
        if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
            # Only keep those still present in df
            return [c for c in cols if c in df.columns]
    except Exception:
        pass
    # Fallback: infer numeric columns directly
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _median_heuristic_gamma(X: np.ndarray) -> float:
    """
        Υπολογίζει gamma για RBF kernel με median heuristic.
        X: (n_samples, n_features)
        """
    # pairwise αποστάσεις
    dists = pairwise_distances(X, metric="euclidean")
    # παίρνουμε το άνω τρίγωνο χωρίς τη διαγώνιο
    upper = dists[np.triu_indices_from(dists, k=1)]
    median_dist = np.median(upper)
    gamma = 1.0 / (2 * (median_dist ** 2))
    return gamma


def run_kpca(
    in_csv: Optional[str] = None,
    out_csv: Optional[str] = None,
    params_json: Optional[str] = None,
    n_components: int = 2,
    gamma: Optional[float] = None,
):
    in_csv, out_csv = _default_paths(in_csv, out_csv)
    params_json = _scaler_params_path(in_csv, params_json)

    print(f"Reading: {in_csv}")
    df = pd.read_csv(in_csv)

    # Choose numeric feature columns (prefer those saved by prepare_kpca)
    feat_cols = _load_numeric_columns(params_json, df)
    if not feat_cols:
        raise RuntimeError("No numeric columns found for KPCA.")

    X = df[feat_cols].to_numpy(dtype=float)
    # Any residual NaNs -> zeros (shouldn't happen after normalization, but safe)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if gamma is None:
        gamma = _median_heuristic_gamma(X)
        print(f"Computed gamma via median heuristic: {gamma:.6g}")
    else:
        gamma = float(gamma)
        print(f"Using provided gamma: {gamma:.6g}")

    kpca = KernelPCA(n_components=n_components, kernel="rbf", gamma=gamma, fit_inverse_transform=False)
    X_kpca = kpca.fit_transform(X)

    # Append KPCA components
    comp_cols = [f"kpca_{i+1}" for i in range(n_components)]
    out_df = df.copy()
    for i, col in enumerate(comp_cols):
        out_df[col] = X_kpca[:, i]

    # Save output CSV
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved KPCA result: {out_csv}")

    # Create a simple 2D scatter plot if at least 2 components were produced
    plot_path = None
    if n_components >= 2:
        try:
            plt.figure(figsize=(8, 6))
            x = out_df["kpca_1"].to_numpy()
            y = out_df["kpca_2"].to_numpy()
            # Color by number_ofC if available
            if "compression_ratio" in df.columns:
                c = df["compression_ratio"].to_numpy()
                sc = plt.scatter(x, y, c=c, cmap="viridis", s=40, edgecolors="k", linewidths=0.3)
                cbar = plt.colorbar(sc)
                cbar.set_label("compression_ratio")
            else:
                plt.scatter(x, y, color="tab:blue", s=40, edgecolors="k", linewidths=0.3)
            plt.title(f"Kernel PCA (RBF) — gamma={gamma:.3g}")
            plt.xlabel("KPCA 1")
            plt.ylabel("KPCA 2")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.splitext(out_csv)[0] + "__kpca_scatter.png"
            plt.savefig(plot_path, dpi=150)
        except Exception as e:
            # If plotting fails (e.g., headless env), continue without blocking
            print(f"Warning: Could not generate KPCA scatter plot: {e}")
        finally:
            try:
                plt.close()
            except Exception:
                pass

    # Save metadata JSON next to output
    meta_path = os.path.splitext(out_csv)[0] + "__kpca_params.json"
    meta = {
        "input_csv": in_csv,
        "output_csv": out_csv,
        "params_json": params_json,
        "feature_columns": feat_cols,
        "n_components": n_components,
        "gamma": gamma,
        "gamma_method": "provided" if gamma is not None else "median_heuristic",
        "plot_path": plot_path,
        "notes": "KPCA with RBF kernel applied to normalized data. Gamma computed via median pairwise distance unless provided. A 2D scatter plot of the first two components is saved if available.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved KPCA params: {meta_path}")


def main():
    # CLI: python data/run_kpca.py [input_csv] [output_csv] [n_components] [gamma]
    # If gamma is omitted or 'auto', the median heuristic is used.
    in_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    ncomp_arg = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 2
    gamma_arg = sys.argv[4] if len(sys.argv) > 4 else None
    if gamma_arg is not None and gamma_arg.lower() == "auto":
        gamma = None
    elif gamma_arg is not None:
        try:
            gamma = float(gamma_arg)
        except Exception:
            print(f"Invalid gamma '{gamma_arg}', falling back to auto.")
            gamma = None
    else:
        gamma = None

    run_kpca(in_arg, out_arg, None, ncomp_arg, gamma)


if __name__ == "__main__":
    main()
