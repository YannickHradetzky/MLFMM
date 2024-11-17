In [[Understanding and Creating Features]] we discussed ways of training non linear models ([[Linear Classification]], [[Linear Classification II - Objective Functions]] and [[Linear Regression]]) with linear algorithms by mapping samples $x \in R^{d}$ to $\phi(x) \in R^{p}$ with usually $p\gg d$. And then we apply the linear model in $p$ dimensions. 
$\phi(x)$ can contain polinomial and cross terms of features in $x$

**The problem:** If $d$ is already large then $p$ is extremely large. 

**The Idea:** We form high dimensional feature vectors only implicitly and **only use the inner product** $\bra{\phi(\vec{x})}\ket{\phi(\vec{x}')}$.

So for $x = (x_{1},x_{2})^{T}$ the second order polynomial would be $\phi(\vec{x})=(1,x_{1},x_{2},\sqrt{ 2 }x_{1}x_{2},x_{1}^{2},x_{2}^{2})^{T}$

Then the inner product can be written as: 

$$
\vec{\phi(\vec{x})}\cdot  \vec{\phi(\vec{x}')} = 1 + \underbrace{ x_{1}x_{1}^{'} + x_{2}x_{2}' }_{ \vec{x}\cdot \vec{x'} }+\underbrace{ 2x_{1}x_{2}x_{1}'x_{2}' + x_{1}^{2}x_{1}'^{2}+x_{2}^{2}x_{2}^{'2} }_{ (\vec{x}\cdot \vec{x'})^{2} }
$$

# Kernel Methods 

## Kernel perceptron 

Instead of doing the update after every Step we can rewrite the final weights and bias as: 

$$
\vec{w} = \sum_{i=1}^{n}\alpha_{i}y^{(i)}\vec{\phi(\vec{x^{(i)}})}
$$
Where $\alpha_{i}$ is the number of times we have been wrong on sample $i$
$$
b = \sum_{i=1}^{n}\alpha_{i}y^{(i)}
$$


**Our Goal:** Determine the $\alpha_{i}$ without having to form $\vec{w}$ because that might be high dimensional. 

The scalar product of weights and features is: 

$$
\begin{align}
\vec{w}\cdot\vec{\phi}(\vec{x}) = \sum_{i=1}^{n}\alpha_{i}y^{(i)}\underbrace{ \vec{\phi}(\vec{x^{(i)}})\vec{\phi}(\vec{x}) }_{ k(\vec{x},\vec{x'}) }
\end{align}
$$
Where $k(\vec{x},\vec{x'})$ is the kernel function. It is a function of two arguments and is always defined as the inner product of feature vectors corresponding to the input argument.


### Algorithm

```python
import numpy as np

def kernel(x1,x2):
	""" Returns the inner product"""
	pass

def get_kernel_matrix(samples, lables)
	"""Returns the kernel matrix where K_ij = y_i * k(x_i,x_j)"""
	pass

def KernelPeceptron(samples, labels, T):
	alpha = np.zeros(samples.shape[0])
	for t in range(T):
		for k in range(samples.shape[0]):
			# Check if you really take a row or column of the matrix
			if labels[k]* alpha * kernel_matrix[:,k] <= 0
				alpha[k] += 1

	return alpha
```

This algorithm can now be run with any valid kernel $k(\vec{x},\vec{x'})$ and usually only a few $\alpha_{i}$ will be non-zero. (Similar to Support-Vector Machine which can be reformulated into a kernel method)

So we reformulated a linear problem with weights $\vec{w}$ for linear features $\vec{\phi}$ with weights per training sample $\alpha_{i}$ by using a non-linear kernel. 


## Kernel Ridge Regression 

**Derivation of analytical solution**
Take away: Predicting a new points expensive because weight is defined as sum over all data points. 

# Kernel Functions

A Kernel function is valid if and only if there exists some feature mapping $\vec{\phi}(\vec{x})$ such that $k(\vec{x},\vec{x'}) = \vec{\phi}(\vec{x})\cdot  \vec{\phi}(\vec{x'})$.

Look at kernel rules and different Kernels in [[MachineLearningForMoleculesAndMaterials.pdf]]

## Gaussian (RBF) Kernel

$$
k(\vec{x}, \vec{x'}) = e^{- \frac{\lvert \lvert \vec{x}-\vec{x'} \rvert  \rvert _{2}}{2\sigma^{2}} }
$$

Exponential sum is an infinite sum (kernel of infinite dimensional feature vectors)
- Any distinct set of training samples is always separable. 
	- Kernel perceptron with the RBF Kernel **always converges**

$$
\begin{align}
k(\vec{x}, \vec{x'})  & = e^{- \frac{\lvert \lvert \vec{x}-\vec{x'} \rvert  \rvert _{2}}{2} }  \\
 & = e^{\lvert \lvert \vec{x} \rvert  \rvert ^{2}_{2}/2} \cdot e^{\vec{x}\cdot  \vec{x'}} \cdot e^{\lvert \lvert \vec{x'} \rvert  \rvert ^{2}_{2}/2} \\
 & =f(\vec{x}) e^{\vec{x} \cdot \vec{x'}} f(\vec{x'})
\end{align}
$$








