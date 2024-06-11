using MultivariateStats
using Plots
using Random
using Statistics

# 1. Gerar um conjunto de dados sintético
Random.seed!(42)  # Para reprodutibilidade
n_samples = 300
n_features = 5
n_centers = 3

# Função para criar blobs de dados
function make_blobs(n_samples, n_features, n_centers)
    centers = randn(n_centers, n_features) * 10
    X = vcat([randn(n_samples ÷ n_centers, n_features) .+ centers[i, :] for i in 1:n_centers]...)
    y = repeat(1:n_centers, inner=n_samples ÷ n_centers)
    return X, y
end

X, y = make_blobs(n_samples, n_features, n_centers)

# 2. Aplicar PCA para reduzir a dimensionalidade para 2 componentes principais
pca_model = fit(PCA, X; maxoutdim=2)
X_pca = transform(pca_model, X)

# 3. Visualizar os resultados
scatter(X_pca[:, 1], X_pca[:, 2], group=y, legend=:topright,
        xlabel="Componente Principal 1", ylabel="Componente Principal 2",
        title="Projeção PCA dos Dados", palette=:viridis)

# 4. Explicar a variância acumulada
explained_variance = principalvars(pca_model)
total_variance_explained = sum(explained_variance[1:2]) / sum(explained_variance)
println("Variância explicada pelo primeiro componente principal: $(explained_variance[1] / sum(explained_variance) * 100):.2f%")
println("Variância explicada pelo segundo componente principal: $(explained_variance[2] / sum(explained_variance) * 100):.2f%")
println("Variância total explicada pelos dois primeiros componentes principais: $(total_variance_explained * 100):.2f%")

