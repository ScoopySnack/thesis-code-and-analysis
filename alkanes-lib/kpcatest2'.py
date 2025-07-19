import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Για 3D plotting
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# Φόρτωση δεδομένων
df = pd.read_csv("alkanes_final.csv")

# Αφαίρεση μη αριθμητικών / αχρήστων στηλών
X = df.drop(columns=['density'], errors='ignore')

# Επιλογή αριθμητικών χαρακτηριστικών
X_numeric = X.select_dtypes(include=['number'])

# Αντικατάσταση ελλιπών τιμών με μέση τιμή
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numeric)

# Κανονικοποίηση χαρακτηριστικών
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Kernel PCA με RBF kernel
kPca = KernelPCA(n_components=3, kernel='rbf', gamma=0.13)
X_kPca = kPca.fit_transform(X_scaled)

# 3D Γράφημα
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_kPca[:, 0], X_kPca[:, 1], X_kPca[:, 2], s=60, edgecolor='k')

ax.set_title('Kernel PCA (RBF) on Alkanes')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.tight_layout()
plt.show()
