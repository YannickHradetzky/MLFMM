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
