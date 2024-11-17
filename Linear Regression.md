s#MLFMM 
The difference between **linear classification** and **linear regression** is working with continuous labels instead of binary ones like in [[Linear Classification II - Objective Functions]] and [[Linear Classification]].

A linear regression function is simply a linear function of the feature vectors. i.e:
$$
f(\vec{x};\vec{w},b) = \vec{w}\cdot\vec{x}+b
$$
Together with different parameters $\theta$ they form a set of functions $\mathcal{F}$.
# Empirical risk and the least square criterion

As in the classification setting, we will measure training error in terms of the **average loss** or **empirical risk**. 

$$
R_{n}(\theta)=\frac{1}{n}\sum_{i=1}^{n}\text{Loss}(\underbrace{ y^{(i)}-f(\vec{x}^{(i)};\vec{w},b) }_{ \text{Residual }z  })
$$
So $\text{Loss}(z)=\frac{z^{2}}{2}$ where the $\frac{1}{2}$ is just there for an easier derivative

Therefore the **Objective Function** becomes:
$$
J_{n}(\theta)=\frac{1}{n}\sum_{i=1}^{n} \frac{(y^{(i)}-(\vec{w}\vec{x}^{(i)}+b))^{2}}{2}
$$

**During training we minimize $R_{n}(\theta)$**

## Reasons for large $R_{n'}^{\text{test}}$:
- estimation errors
	- $S_{n}$ was too noisy
		- hard to find estimation of $\theta$
- structural errors
	- relationship between $y^{(i)}$ and $\vec{x}^{(i)}$ is not linear
		- increase flexibility in the model in $\mathcal{F}$
			- Makes it harder to find estimation of $\theta$

## Optimizing the least square cirterion 

### Stochastic Gradient Descent
#TODO Check the formulas

$$
\begin{align}
\nabla_w \frac{(y^{(t)} - (w \cdot x^{(t)} + b))^2}{2}  & = (y^{(t)} - (w \cdot x^{(t)} + b)) \nabla_w (y^{(t)} - (w \cdot x^{(t)} + b))  \\
 & = - (y^{(t)} - (w \cdot x^{(t)} + b)) x^{(t)}
\end{align}
$$

$$
\begin{align}
\nabla_b \frac{(y^{(t)} - (w \cdot x^{(t)} + b))^2}{2}  & = (y^{(t)} - (w \cdot x^{(t)} + b)) \nabla_b (y^{(t)} - (w \cdot x^{(t)} + b))  \\
 & = - (y^{(t)} - (w \cdot x^{(t)} + b))
\end{align}
$$

#### Algorithm 
```python
import random

def linear_reg_sgd(data, T, eta):
    # Initialize weights and bias
    w = 0  # Assuming w is a scalar for simplicity; if it's a vector, initialize as np.zeros(len(data[0][0]))
    b = 0  # Bias term

    # SGD iteration
    for t in range(1, T + 1):
        # Select a random data point (x_i, y_i)
        x_i, y_i = random.choice(data)
        
        # Update w and b
        prediction = w * x_i + b
        error = y_i - prediction

        w = w + eta * error * x_i
        b = b + eta * error

    return w, b
```

### Closed form Solution 

Try to minimize $J_{n}(\theta)$ directly by setting the gradient to zero
(5.9 in [[MachineLearningForMoleculesAndMaterials.pdf]])

**Pseudo Code**
```python
import numpy as np
def phi(x):
	""" creates the features"""
	return x
	
x = [phi(x) for x in data]
y = [y for y in labels]

c = np.dot(x.T,y)
A = np.dot(x,x.T)

A_inv = np.inv(A)
theta = np.dot(A_inv,c)

```

# Form of Regularization

The regularization acts as a bias towards a default answer or value. This resistance helps to ensure that our predictions generalize well. 

**Optimize for the simples answer when evidence is weak or absent**

## $L_{2}$ Regularization

This ensures that none of the features gets preferred over another to provide a good generalization.

$$
J_{n,\lambda} = \frac{\lambda}{2}\lvert \lvert \vec{w} \rvert  \rvert _{2}^{2} + R_{n}(\theta)=\frac{\lambda}{2}\lvert \lvert \vec{w} \rvert  \rvert _{2}^{2}+\frac{1}{2n}\sum_{i=1}^{n}(y^{i}-(\vec{w}\vec{x}+b))^{2}
$$
This is know as **Ridge Regression**

```python
import numpy as np

def phi(x):
    # Transform the input feature x as needed
    return x  # For simplicity, we use the identity function. Modify as needed.

def ridge_regression(data, T, lambda_param, eta):
    """
    Perform Ridge Regression using the stochastic gradient descent approach.

    Parameters:
    - data: A list of tuples where each tuple contains (x(i), y(i)).
    - T: The number of iterations.
    - lambda_param: The regularization parameter.
    - eta: The learning rate.

    Returns:
    - w: The weight vector.
    - b: The bias term.
    """
    n = len(data)
    w = np.zeros_like(phi(data[0][0]))  # Initialize weights
    b = 0  # Initialize bias

    for t in range(T):
        # Select i ∈ {1, ..., n} at random
        i = np.random.randint(n)
        x_i, y_i = data[i]
        # Compute the predicted value
        prediction = np.dot(w, phi(x_i)) + b
        # Update weights
        w = (1 - lambda_param * eta) * w + eta * (y_i - prediction) * phi(x_i)
        # Update bias
        b += eta * (y_i - prediction)
    return w, b
```

## $L_{1}$ Regularization

$$
J_{n,\lambda} = \frac{\lambda}{2}\lvert \lvert \theta \rvert  \rvert _{1} + R_{n}(\theta)=\frac{\lambda}{2}\lvert \lvert \theta \rvert  \rvert _{1}+\frac{1}{2n}\sum_{i=1}^{n}(y^{i}-(\vec{w}\vec{x}+b))^{2}
$$Since the $\lvert \lvert \dots \rvert \rvert_{1}$ norm simply sums the elements of our parameter vector $\vec{\theta}$ this objective functions will optimize for using as little parameters as possible. So if one feature is not needed then it's parameter will be less and less significant or even be $0$. This yields a kind of **automatic feature selection** 

## $L_{1}$ and $L_{2}$ 

This is referred to as **Elastic Net**.


# Effects of Regularization

The regularization term shifts the emphasis away from the training data. As a result, we expect that higher values of λ will have a negative impact on the training error.

- Larger $\lambda$ leads to lower generalization error (lower $R^{\text{test}}_{n'}$)
- at some point $\lambda$ is to large, and we don't fit/learn anything from the data.

 