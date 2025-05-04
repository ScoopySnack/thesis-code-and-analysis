import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Sample data, replace with actual data
data = {
    'C_atoms': [1, 2, 3, 4, 5],
    'H_atoms': [4, 6, 8, 10, 12],
    'Molecular_Weight': [16.04, 30.07, 44.10, 58.12, 72.15],
    'Boiling_Point': [-161.5, -88.6, -42.1, 0.5, 36.1],
    'Density': [0.656, 1.356, 2.009, 2.493, 3.050]  # Sample fake values
}
df = pd.DataFrame(data)

# print(df)

# drop('Density', axis=1)
# Step 2: Feature matrix and target, assuming 'Density' is the target variable
X = df  # Features

# Step 3: Scale features, important for KPCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply KPCA, using RBF kernel
kpca = KernelPCA(n_components=3, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X_scaled)

# Step 5: Plot with Density as color
plt.figure(figsize=(8, 6))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], s=100, edgecolors='k')
# plt.colorbar(label="Density (g/mL)")
plt.title("Kernel PCA of Alkanes")
plt.xlabel("PC1") # First Principal Component is the first axis
plt.ylabel("PC2") # Second Principal Component is the second axis
plt.grid(True)
plt.show()
