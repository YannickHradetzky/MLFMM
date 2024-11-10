#MLFMM

# Why machine learning
- Accelerating Discoveries
- Data Analysis, Pattern recognition 
- Predictive Modelling 
- Adaptability

# Branches of machine Learning 
- unsupervised learning
	- unlabelled data
	- feature extraction or reduction
	- clustering
- supervised learning
	- labelled data
	- regression 
		- Find a function or relation between data and floating point values
	- categorical data -> classification 
- reinforcement learning
	- "gamification"
	- feedback for the algorithm
- generative models fall between unsupervised- and supervised learning 

# Chapter 0 - Recap
## Linear Algebra
- $\underline{v}$ column vector in $R^{d}$
- $v^{T}$ row vector
- a vector can be scaled by a scalar $a$ : $a\underline{v}$
- the dot product of two vector of the same length : $w^{T}\underline{v}=\sum_{i=1}^{d}w_{i}v_{i}$
- addition / subtraction 
- length of a vector (norm) : $||v||_{2}=\sqrt{ \sum_{i=1}^{d}v_{i}^{2} }$
- a matrix is a rectangular vector of values
- matrix addition/subtraction is element wise : $(A+B)_{ij}=a_{ij}+b_{ij}$
- matrix multiplication 
	- $$
\mathbf{C}_{ij} = \sum_{k=1}^{n} \mathbf{A}_{ik} \mathbf{B}_{kj}, \quad \text{for } i = 1, 2, \dots, m \text{ and } j = 1, 2, \dots, p.
$$
- matrix transpose
	- simply swapping the rows and columns
	- $$\mathbf{A}^T_{ij} = \mathbf{A}_{ji}$$
- identity matrix
- matrix inverse
	- $$\mathbf{A} \mathbf{A}^{-1} = \mathbf{I}$$
- Eigenvalues
	- $$\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$$
	- to calculate them : $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$





## Analysis
### functions :  are maps from input to output space
$$f(x):=\mathbf{R}^{d}\to \mathbf{R}^{p}$$
### derivatives
measure of change of a function with respect to an input parameter
### partial derivatives
measure of change w.r.t each variable without considering for possible interdependencies
### gradient

$$\nabla f = \begin{pmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{pmatrix}$$
### chain rule : 
if $y=f(g(x))$
$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$
### Jacobian Matrix

The Jacobian matrix is a matrix of all first-order partial derivatives of a vector-valued function. If you have a vector-valued function $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$ defined as:
$$

\mathbf{f}(\mathbf{x}) = \begin{pmatrix}

f_1(x_1, x_2, \ldots, x_n) \

f_2(x_1, x_2, \ldots, x_n) \

\vdots \

f_m(x_1, x_2, \ldots, x_n)

\end{pmatrix}

$$
where $\mathbf{x} = (x_1, x_2, \ldots, x_n)$, the Jacobian matrix $\mathbf{J}$ of the function $\mathbf{f}$ is defined as:
$$

\mathbf{J} = \begin{pmatrix}

\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \

\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \

\vdots & \vdots & \ddots & \vdots \

\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}

\end{pmatrix}

$$
Each element of the Jacobian matrix $J_{ij}$ is given by:

$$

J_{ij} = \frac{\partial f_i}{\partial x_j}

$$
The Jacobian matrix captures how the vector-valued function $\mathbf{f}$ changes with respect to the input vector $\mathbf{x}$ and is essential in multivariable calculus, particularly in optimization, numerical methods, and differential equations.

### Hessian Matrix

The Hessian matrix is a square matrix of second-order partial derivatives of a scalar-valued function. If you have a scalar-valued function , defined as:

$$

f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n)

$$
where , the Hessian matrix  of the function  is defined as:

  

$$

\mathbf{H} = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \

\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \

\vdots & \vdots & \ddots & \vdots \

\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}

\end{pmatrix} = J(\Delta f)
$$

Each element of the Hessian matrix  is given by:

$$

H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}

$$
The Hessian matrix provides information about the local curvature of the function  and is particularly useful in optimization problems, as it helps determine whether a critical point is a local minimum, local maximum, or a saddle point.

### Taylor series
The Taylor series is a representation of a function as an infinite sum of terms calculated from the values of its derivatives at a single point. If a function  is infinitely differentiable at a point , its Taylor series around that point is given by:

$$
f(x) = f(a) + f’(a)(x - a) + \frac{f’’(a)}{2!}(x - a)^2 + \frac{f’’’(a)}{3!}(x - a)^3 + \cdots
$$
This can be expressed more compactly using summation notation as:

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x - a)^n
$$
where:

•  $f^{(n)}(a)$  is the  $n$ -th derivative of  $f$  evaluated at  $a$ 
• $n!$ is the factorial of $n$
•  $(x - a)^n$  represents the  $n$ -th power of the difference between  $x$  and  $a$ .

The Taylor series can be used to approximate functions near the point , and if the series converges to the function for all , the function is said to be equal to its Taylor series in that interval.

## Probability/Statistics
### Expectation Value of Random Variable X

#### Discrete Variables

The expectation value (or expected value) of a discrete random variable X is defined as:

$$
E[X] = \sum_{i} x_i P(X = x_i)
$$

where x_i are the possible values of X and P(X = x_i) is the probability mass function of X.

#### Continuous Variables

The expectation value of a continuous random variable X is defined as:

$$
E[X] = \int_{-\infty}^{\infty} x f_X(x) , dx
$$

where f_X(x) is the probability density function of X.

#### Empirically

The empirical expectation value based on sample data x_1, x_2, \ldots, x_n is calculated as:

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

where n is the number of observations, and \bar{x} is the sample mean, which serves as an estimate of the population expectation value.

Here are the definitions for variance and covariance:
### Variance

The variance of a random variable  measures the spread of its values around the expected value . It is defined as:

#### For Discrete Variables:
$$
\text{Var}(X) = E[(X - E[X])^2] = \sum_{i} (x_i - E[X])^2 P(X = x_i)
$$
#### For Continuous Variables:
$$
\text{Var}(X) = E[(X - E[X])^2] = \int_{-\infty}^{\infty} (x - E[X])^2 f_X(x) , dx
$$
### Covariance

Covariance measures the degree to which two random variables  and  change together. It is defined as:
#### For Discrete Variables:
$$
\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = \sum_{i} \sum_{j} (x_i - E[X])(y_j - E[Y]) P(X = x_i, Y = y_j)
$$
#### For Continuous Variables:
$$

\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} (x - E[X])(y - E[Y]) f_{X,Y}(x, y) , dx , dy

$$

where  is the joint probability density function of  and .

## Summary of Notation
### training example

$\underline{x}^{(i)}$ with feature vector  $\underline{\phi(x^{(i)})}$
### label

$y^{(i)}$, usually scalars

### labeled data

Is a set of tuples $S_{n= }\{ \underline{x^{(i)}}, y^{(i)} \}_{i=1}^{n}$

### unlabeled data 

Is a set of features : $S_{n}=\{ \phi(x^{(i)}) \}_{i=1}^{n}$

### Data generation process

Unseen process that produces the labels for a given $\underline{x^{(i)}}$

### Model

A function $\hat{f}(x)$ that a returns a prediction $\hat{y}$

### Parameters

The entire parameters of a model are denoted with $\theta=\Theta$. If we have a linear model, $\underline{w}$ weights and $b$ the bias

