import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons


#generate data
# make_moons is a function that generates a two-dimensional dataset with a crescent moon shape
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

print(X)
print(y)

#Apply Kernel PCA
# rbf kernel with gamma=15, rbf kernel is a popular choice for non-linear data
# gamma is a parameter that defines how far the influence of a single training example reaches
kPca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kPca = kPca.fit_transform(X)

#Plot the results
plt.figure(figsize=(8, 6))
# x_kPca[:, 0] and x_kPca[:, 1] are the first and second principal components, respectively
# c=y indicates the color of the points based on their labels
# cmap='viridis' is a colormap that provides a gradient of colors
# edgecolor='k' adds a black edge around the points
# s=50 sets the size of the points
plt.scatter(X_kPca[:, 0], X_kPca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Kernel PCA with RBF Kernel')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()