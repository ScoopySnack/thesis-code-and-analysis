import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


df = pd.read_csv("numeric_only_normalized.csv")

# π.χ. target
y = df["Tc"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

def spectral_gap_for_gamma(X_scaled, gamma, top_k=None):
    # Gram
    K = rbf_kernel(X_scaled, X_scaled, gamma=gamma)
    # Centering
    kc = KernelCenterer()
    Kc = kc.fit_transform(K)
    # Eigenvalues (συμμετρικός πίνακας)
    w = np.linalg.eigvalsh(Kc)
    w = np.sort(w)[::-1]   # φθίνουσα
    if top_k:
        w = w[:top_k]
    diffs = np.abs(np.diff(w))
    gap = diffs.max() if len(diffs) else 0.0
    gap_idx = diffs.argmax() + 1 if len(diffs) else 1
    return gap, gap_idx, w

gammas = np.logspace(-3, 1, 30)  # π.χ. 0.001 έως 10
results = []
for g in gammas:
    gap, d_est, eigvals = spectral_gap_for_gamma(X_scaled, g)
    results.append((g, gap, d_est))

# επίλογη
g_star, gap_star, d_star = max(results, key=lambda t: t[1])
print(f"Best gamma={g_star:.5g}, spectral gap={gap_star:.3f}, intrinsic d≈{d_star}")

n_components = max(2, d_star + 1)  # π.χ. 2–5
kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=g_star, fit_inverse_transform=False)
Z = kpca.fit_transform(X_scaled)   # embedding [N, n_components]

# 2D scatter
plt.figure(figsize=(6,5))
sc = plt.scatter(Z[:,0], Z[:,1], c=y, s=35, edgecolors='k', linewidths=0.3)
plt.xlabel("KPCA 1"); plt.ylabel("KPCA 2"); plt.title("KPCA embedding")
plt.colorbar(sc, label="Enthalpy of combustion")
plt.tight_layout(); plt.show()

# Distance-preservation (Spearman μεταξύ αποστάσεων)
D_X = squareform(pdist(X_scaled))
D_Z = squareform(pdist(Z))
rho, _ = spearmanr(D_X.ravel(), D_Z.ravel())
print(f"Distance monotonicity (Spearman) ≈ {rho:.3f}")


gap, d_est, eigvals = spectral_gap_for_gamma(X_scaled, g_star)
plt.figure(figsize=(6,4))
plt.plot(eigvals, marker='o')
plt.yscale('log'); plt.xlabel('Index'); plt.ylabel('Eigenvalue (log)')
plt.title(f'Kernel eigen-spectrum (gamma={g_star:.3g})')
plt.tight_layout(); plt.show()
print(f"Estimated intrinsic dimension (gap index): {d_est}")

