import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import pandas as pd


df = pd.read_csv("alkanes_predicted.csv")

X=df
# Assuming 'Density' is the target variable, we will drop it for KPCA
targets_to_predict = X.drop(columns=['Density','name','smiles'], errors='ignore')  # Drop 'Density' if it exists
# If 'Density' is not in the DataFrame, it will not raise an error due to errors='ignore'
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # Initialize the scaler
targets_to_predict = scaler.fit_transform(targets_to_predict)  # Scale the features
# Apply Kernel PCA
kPca = KernelPCA(n_components=2, kernel='rbf', gamma=15)  # RBF kernel with gamma=15
X_kPca = kPca.fit_transform(targets_to_predict)  # Transform the data
# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_kPca[:, 0], X_kPca[:, 1], s=50, edgecolors='k')  # Scatter plot of the first two principal components
plt.title('Kernel PCA with RBF Kernel on Alkanes Data')
plt.xlabel('Principal Component 1')  # Label for the x-axis
plt.ylabel('Principal Component 2')  # Label for the y-axis
plt.grid(True)  # Add grid for better readability
plt.show()  # Display the plot
# This code applies Kernel PCA to the alkanes dataset and visualizes the results.


