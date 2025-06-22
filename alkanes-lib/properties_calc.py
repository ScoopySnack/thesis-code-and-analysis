import json
from rdkit import Chem
import networkx as nx
import numpy as np

# ---------- SMILES MAP ----------
smiles_map = {
    "methane": "C",
    "ethane": "CC",
    "propane": "CCC",
    "butane": "CCCC",
    "isobutane": "CC(C)C",
    "pentane": "CCCCC",
    "isopentane": "CC(C)CC",
    "2,2-dimethylpropane": "CC(C)(C)C",
    "hexane": "CCCCCC",
    "2-methylpentane": "CC(C)CCC",
    "3-methylpentane": "CCC(C)CC",
    "2,2-dimethylbutane": "CC(C)(C)CC",
    "3,3-dimethylbutane": "CCC(C)(C)C",
    "heptane": "CCCCCCC",
    "2-methylhexane": "CC(C)CCCC",
    "3-methylhexane": "CCC(C)CCC",
    "2,2-dimethylpentane": "CC(C)(C)CCC",
    "3,3-dimethylpentane": "CCC(C)(C)CC",
    "2,3-dimethylpentane": "CC(C)C(C)CC",
    "2,4-dimethylpentane": "CC(C)CC(C)C",
    "3-ethylpentane": "CCC(CC)CC",
    "2,2,3-trimethylbutane": "CC(C)(C)C(C)C",
    "octane": "CCCCCCCC"
}

# ---------- GRAPH CALCULATION ----------
def smiles_to_nx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G

def compute_graph_properties(smiles):
    G = smiles_to_nx(smiles)
    A = nx.to_numpy_array(G)
    L = nx.laplacian_matrix(G).toarray()

    eigvals_A = np.linalg.eigvalsh(A)
    eigvals_L = np.linalg.eigvalsh(L)

    perron = float(np.max(eigvals_A))
    fiedler = float(sorted(eigvals_L)[1]) if len(eigvals_L) > 1 else 0.0

    degrees = [d for _, d in G.degree()]
    total_deg = sum(degrees)
    probs = [d / total_deg for d in degrees if d > 0]
    entropy = -sum(p * np.log2(p) for p in probs)

    return {
        "perron_frobenius": round(perron, 4),
        "fiedler_eigenvalue": round(fiedler, 4),
        "compression_ratio": None,
        "information_content": round(entropy, 4)
    }

# ---------- LOAD DATA ----------
with open("alkanes_cleaned.json", "r") as file:
    data = json.load(file)

for name, props in data["alkanes"].items():
    if name in smiles_map:
        smiles = smiles_map[name]
        props["smiles"] = smiles
        props["graph_properties"] = compute_graph_properties(smiles)

# ---------- SAVE ENRICHED FILE ----------
with open("alkanes_final.json", "w") as out:
    json.dump(data, out, indent=2)

print("✅ Done! Saved as 'alkanes_final.json'")
