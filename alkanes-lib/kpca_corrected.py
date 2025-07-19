from math import gamma

import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

df = pd.read_csv("alkanes_final.csv")

# Drop non-numerical or irrelevant columns
X = df.drop(columns=['density'], errors='ignore')

# Keep only numeric features
X_numeric = X.select_dtypes(include=['number'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numeric)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

A = eigvals(X_scaled.T @ X_scaled)  # Eigenvalues of the covariance matrix
# Print eigenvalues in paired format
print("Eigenvalues of the covariance matrix:")
for i in range(0, len(A), 2):
    if i + 1 < len(A):
        print(f"Pair {i//2 + 1}: {A[i]:.4f}, {A[i+1]:.4f}")
    else:
        print(f"Single value {i//2 + 1}: {A[i]:.4f}")

#Calculate gamma based on the eigenvalues
gamma_value = 1 / (2 * A.mean())
# Print calculated gamma value
print(f"Calculated gamma value: {gamma_value:.4f}")


# Apply Kernel PCA
kPca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma_value)  # Adjust gamma if needed
X_kPca = kPca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_kPca[:, 0], X_kPca[:, 1], s=50, edgecolors='k')
plt.title('Kernel PCA with RBF Kernel on Alkanes Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.grid(True)
plt.show()
