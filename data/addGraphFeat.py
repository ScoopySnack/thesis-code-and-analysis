import os
import sys
import math
import pandas as pd
from rdkit import Chem

# Ensure project root on sys.path for package imports when run from subfolders
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alkanes_data.graph_features import mol_to_graph, get_graph_features


def _sig(x, sig=3):
    """Round to given significant digits; return None for non-finite/invalid."""
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return None
        return float(f"{xf:.{sig}g}")
    except Exception:
        return None


def compute_graph_descriptors(smiles: str):
    """Compute requested graph descriptors for a molecule SMILES.

    Returns a dict with keys:
      - pf_eigenvalue
      - fiedler_eigenvalue
      - degree_entropy
      - compression_ratio
    """
    if not isinstance(smiles, str) or smiles.strip() == "":
        return {
            "pf_eigenvalue": None,
            "fiedler_eigenvalue": None,
            "degree_entropy": None,
            "compression_ratio": None,
        }
    try:
        # Validate SMILES can be parsed before building graph
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        G = mol_to_graph(smiles)
        pf, fiedler, entropy, comp = get_graph_features(G)
        return {
            "pf_eigenvalue": _sig(pf),
            "fiedler_eigenvalue": _sig(fiedler),
            "degree_entropy": _sig(entropy),
            "compression_ratio": _sig(comp),
        }
    except Exception:
        return {
            "pf_eigenvalue": None,
            "fiedler_eigenvalue": None,
            "degree_entropy": None,
            "compression_ratio": None,
        }


def main(in_csv: str = None, out_csv: str = None):
    # Default paths
    data_dir = os.path.join("data")
    default_in = os.path.join(data_dir, "alkanes_core_with_smiles_final.csv")
    if in_csv is None:
        in_csv = default_in
    if out_csv is None:
        base, ext = os.path.splitext(in_csv)
        out_csv = base + "_with_graph.csv"

    print(f"Reading: {in_csv}")
    df = pd.read_csv(in_csv)

    # Expect a SMILES column named 'SMILES'
    smiles_col = None
    for cand in ["SMILES", "smiles", "Smiles"]:
        if cand in df.columns:
            smiles_col = cand
            break
    if smiles_col is None:
        raise KeyError("Input CSV must contain a 'SMILES' column")

    # Compute descriptors row-wise
    records = []
    for s in df[smiles_col].tolist():
        records.append(compute_graph_descriptors(str(s) if pd.notna(s) else ""))
    feat_df = pd.DataFrame.from_records(records)

    # Concatenate and save
    out_df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved with graph descriptors: {out_csv}")


if __name__ == "__main__":
    # CLI usage: python data/addGraphFeat.py [input_csv] [output_csv]
    in_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(in_arg, out_arg)
