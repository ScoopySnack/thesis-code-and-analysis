import networkx as nx
import numpy as np
from rdkit import Chem


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    adj = Chem.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)
    return G


def get_graph_features(G):
    """
    Compute graph descriptors as specified:
    - Perron–Frobenius eigenvalue: largest eigenvalue of adjacency matrix.
    - Fiedler eigenvalue: second-smallest eigenvalue of Laplacian (algebraic connectivity).
    - Graph entropy: Shannon entropy (natural log) of the degree distribution.
    - Compression ratio: N^2 / max(2E, 1), comparing adjacency matrix entries vs. edge-list size.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Adjacency matrix and Perron–Frobenius eigenvalue
    A = nx.to_numpy_array(G, dtype=float)
    eigs_A = np.linalg.eigvalsh(A)  # symmetric -> real eigenvalues
    pf = float(np.max(eigs_A))

    # Laplacian and Fiedler eigenvalue (avoid SciPy): L = D - A
    degs = A.sum(axis=1)
    L = np.diag(degs) - A
    eigs_L = np.linalg.eigvalsh(L)
    if n > 1:
        fiedler = float(np.sort(eigs_L)[1])  # second-smallest
    else:
        fiedler = 0.0

    # Degree distribution entropy (Shannon, natural log)
    degrees = [d for _, d in G.degree()]
    counts = np.bincount(degrees) if len(degrees) else np.array([0])
    probs = counts[counts > 0] / float(n)
    degree_entropy = float(-(probs * np.log(probs)).sum()) if probs.size > 0 else 0.0

    # Simple compression ratio: adjacency entries vs. edge list entries (undirected -> 2 entries per edge)
    compression_ratio = float((n * n) / max(2 * m, 1))

    return pf, fiedler, degree_entropy, compression_ratio