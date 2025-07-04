import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from .smiles_generator import generate_linear_alkane_smiles
from .graph_features import mol_to_graph, get_graph_features
from .data_fetcher import fetch_properties

def generate_alkane_dataset(min_c=1, max_c=20, save_csv=True, save_json=True):
    data = []
    for n in range(min_c, max_c + 1):
        name = f"n-alkane-C{n}"
        smiles = generate_linear_alkane_smiles(n)
        mol = Chem.MolFromSmiles(smiles)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        G = mol_to_graph(smiles)
        pf, fiedler, entropy, compression = get_graph_features(G)
        props = fetch_properties(smiles)
        row = {
            "name": name,
            "formula": formula,
            "SMILES": smiles,
            **props,
            "graph_properties": {
                "perron_frobenius": pf,
                "fiedler_eigenvalue": fiedler,
                "spectral_entropy": entropy,
                "compression_ratio": compression
            }
        }
        data.append(row)

    if save_csv:
        df = pd.json_normalize(data)
        df.to_csv("alkanes_data.csv", index=False)

    if save_json:
        with open("alkanes_data.json", "w") as f:
            json.dump(data, f, indent=4)