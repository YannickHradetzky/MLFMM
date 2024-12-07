# Principal Component Analysis (PCA)

- widely used technique for dimensionality reduction and feature extraction.
- Transforms data into a new coordinate system such that the greatest variance by any projection of the data lies on the first principal component, the second greatest variance on the second principal component, and so on. 
- Often used as a preprocessing step for other machine learning algorithms to reduce the dimensionality of the data. 

## Algorithm 

1) Standardize the data: Ensure each feature has zero mean. If features have different units also normalize the variance. 
2) Compute covariance matrix
$$
\Sigma=\frac{1}{n-1}\sum_{i=1}^{n}(\vec{x}^{(i)}-\bar{\vec{x}})(\vec{x}^{(i)}-\bar{\vec{x}})^{T} = 1(n-1)X^{T}X
$$
	2.1) Where $X$ is the $n \times d$ matrix of all stacked examples.
3) Compute the eigenvalues and eigenvectors of the covariance matrix
4) Sort the eigenvalues and their corresponding eigenvectors in descending order
5) To reduce the dimensionality: Select the top $k$ eigenvectors to dorm the projection matrix $W$ (Where each column is an eigenvector) and transform the original data
$$
Z = XW
$$
 Typical choices are 80% or 90%. 


# Gaussian Mixture Models (GMM)

Gaussian Mixture Models (GMM) are probabilistic models that assume the data is generated from a mixture of several Gaussian distributions with unknown parameters. Each component in the mixture model represents a cluster. GMMs are used in clustering, density estimation, and as generative models for classification tasks.

A GMM can be defined as: 
$$
p(\vec{x}|\theta)=\sum_{k=1}^{K}\pi_{k}N(\vec{x}|\mu_{k}\Sigma_{k})
$$
where $\pi_{k}$ are the mixing components (ensuring normalisation) and $\mu_{k}$ and $\Sigma_{k}$ are the mean (vector) and covariance (matrix) of the $k$-th Gaussian distribution.

Under the Assumption (constraint) that $\Sigma$ is diagonal, this simplifies to the product of one-dimensional Gaussians, one for each of the d dimensions

```python 
def gmm_algorithm(data, K, max_iter):
    # Initialize model parameters
    means = initialize_means(data, K)
    covariances = initialize_covariances(data, K)
    weights = initialize_weights(K)
    
    for iter in range(max_iter):
        # Expectation step
        posterior_probabilities = calculate_posterior_probabilities(
						        data,
						         means,
						         covariances,
						         weights)
        
        # Maximization step
        means, covariances, weights = update_model_parameters(posterior_probabilities, data)
    
    return means, covariances, weights
```


# Heuristics for Analyzing Clustering Results

## Silhouette Score

The Silhouette Score is an internal validation metric used to measure the quality of a clustering. It quantifies how well each point is clustered, taking into account both cohesion (how close a point is to other points in the same cluster) and separation (how far a point is from points in other clusters). For each data point i, the Silhouette coefficient s(i) is defined as:

$$
s(i) =  \frac{b(i)-a(i)}{\text{max}(a(i),b(i))}
$$
- a(i) is the average distance between point i and all other points in the same cluster (intra- cluster distance).
- b(i) is the minimum average distance from point i to all points in the nearest different cluster (inter-cluster distance).

### Interpretation 
- s(i) ≈ 1: The point is well-clustered, with its cluster being dense and well-separated from others.
- s(i) ≈ 0: The point is on or near the boundary between two clusters.
- s(i) < 0: The point may have been assigned to the wrong cluster.

## Normalized mutual Information 

The Normalized Mutual Information (NMI) is an external validation metric that measures the similarity between two clustering results. It is particularly useful when you have ground truth labels or another clustering result to compare against. 
[[Specific- and Mutual-Information]]

NMI quantifies the amount of information shared between the predicted clustering U and the true clustering V . It is given by:
$$
\text{NMUI} = \frac{2I(U,V)}{H(U)+H(V)}
$$

