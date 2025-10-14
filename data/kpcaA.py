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


gap, d_est, eigvals1 = spectral_gap_for_gamma(X_scaled, g_star)
plt.figure(figsize=(6,4))
plt.plot(eigvals1, marker='o')
plt.yscale('log'); plt.xlabel('Index'); plt.ylabel('Eigenvalue (log)')
plt.title(f'Kernel eigen-spectrum (gamma={g_star:.3g})')
plt.tight_layout(); plt.show()
print(f"Estimated intrinsic dimension (gap index): {d_est}")


def spectral_gaps_plot(X_scaled, gamma, top_k=None, logy=False, title=None):
    """
    Υπολογίζει και κάνει plot τα |λ_{i+1} - λ_i| για τον centered Gram matrix του RBF kernel.
    - X_scaled: (N, D) scaled features
    - gamma: RBF parameter
    - top_k: αν θες να κρατήσεις μόνο τις top_k ιδιοτιμές
    - logy: αν True, κάνει log-scale στον άξονα y για ευκρίνεια
    """
    # 1) Gram + centering
    K = rbf_kernel(X_scaled, X_scaled, gamma=gamma)
    Kc = KernelCenterer().fit_transform(K)

    # 2) Eigenvalues (συμμετρικός πίνακας) σε φθίνουσα σειρά
    eigvals = np.linalg.eigvalsh(Kc)[::-1]
    if top_k is not None:
        eigvals = eigvals[:top_k]

    # 3) Διαφορά διαδοχικών ιδιοτιμών (απόλυτη τιμή)
    diffs = np.abs(np.diff(eigvals))
    idx = np.argmax(diffs)  # θέση μέγιστου gap
    max_gap = diffs[idx]
    intrinsic_d = idx + 1    # εκτίμηση intrinsic dimension

    # 4) Plot
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(np.arange(1, len(diffs)+1), diffs, marker='o', linewidth=1)
    ax.axvline(intrinsic_d, ls='--', color='red', label=f"max gap @ i={intrinsic_d}")
    ax.set_xlabel("i (μεταξύ λ_i και λ_{i+1})")
    ax.set_ylabel("|λ_{i+1} - λ_i|")
    if logy:
        ax.set_yscale('log')
    if title is None:
        title = f"Spectral gaps for RBF γ={gamma:g}"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return {
        "eigvals": eigvals,
        "diffs": diffs,
        "max_gap": float(max_gap),
        "gap_index_i": int(intrinsic_d)  # d ≈ intrinsic dimension
    }

# Παράδειγμα χρήσης:
res = spectral_gaps_plot(X_scaled, gamma=g_star, top_k=80, logy=True)
print(res)