import networkx as nx
import numpy as np
from rdkit import Chem

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    adj = Chem.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)
    return G

def get_graph_features(G):
    A = nx.to_numpy_array(G)
    L = nx.laplacian_matrix(G).todense()
    eigenvalues_A = np.linalg.eigvals(A)
    eigenvalues_L = np.linalg.eigvals(L)
    pf = max(eigenvalues_A).real
    fiedler = sorted(eigenvalues_L)[1].real if len(eigenvalues_L) > 1 else 0
    spectral_entropy = -sum((l/np.sum(eigenvalues_L)) * np.log((l/np.sum(eigenvalues_L)))
                            for l in eigenvalues_L if l > 0)
    compression_ratio = np.count_nonzero(A) / A.size
    return pf, fiedler, spectral_entropy, compression_ratio