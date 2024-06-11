import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# 1. Gerar um conjunto de dados sintético
X, y = make_blobs(n_samples=300, n_features=5, centers=3, random_state=42)

# 2. Aplicar PCA para reduzir a dimensionalidade para 2 componentes principais
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Visualizar os resultados
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Projeção PCA dos Dados')
plt.colorbar(label='Classe')
plt.show()

# 4. Explicar a variância acumulada
explained_variance = pca.explained_variance_ratio_
print(f'Variância explicada pelo primeiro componente principal: {explained_variance[0]:.2f}')
print(f'Variância explicada pelo segundo componente principal: {explained_variance[1]:.2f}')
print(f'Variância total explicada pelos dois primeiros componentes principais: {explained_variance.sum():.2f}')

