import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("numeric_only_normalized.csv")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --- Εφαρμογή Kernel PCA ---
kpca = KernelPCA(n_components=3, kernel='rbf', gamma=2.13)
X_kpca = kpca.fit_transform(X_scaled)

# --- 3D Plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Αν έχεις κάποια στήλη π.χ. "number_ofC" για χρωματισμό:
if "degree_entropy" in df.columns:
    colors = df["degree_entropy"]
    sc = ax.scatter(X_kpca[:,0], X_kpca[:,1], X_kpca[:,2],
                    c=colors, cmap='viridis', s=60, edgecolors='k')
    cbar = plt.colorbar(sc)
    cbar.set_label("Entropy")
else:
    ax.scatter(X_kpca[:,0], X_kpca[:,1], X_kpca[:,2],
               color='tab:blue', s=60, edgecolors='k')

ax.set_title("3D Kernel PCA (RBF kernel)")
ax.set_xlabel("KPCA 1")
ax.set_ylabel("KPCA 2")
ax.set_zlabel("KPCA 3")

plt.show()

out_df = pd.DataFrame(X_kpca, columns=["kpca_1", "kpca_2", "kpca_3"])
out_df.to_csv("output_kpca3d.csv", index=False)