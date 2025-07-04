
import matplotlib.pyplot as plt
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

# Apply Kernel PCA
kPca = KernelPCA(n_components=2, kernel='rbf', gamma=0.001)  # Adjust gamma if needed
X_kPca = kPca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_kPca[:, 0], X_kPca[:, 1], s=50, edgecolors='k')
plt.title('Kernel PCA with RBF Kernel on Alkanes Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
