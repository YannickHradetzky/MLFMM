#MLFMM 
[[Linear Classification]]

In general , if we consider classifiers specified by parameters $\vec{w},b$, e.g. the set of linear classifiers, we first try to turn the learning problem into an optimization problem over these parameters. 

We typically try to minimize : 
$$
J(\vec{w},b)=\underbrace{ \frac{1}{n}\sum_{i=1}^{n}\text{Loss}(\vec{x}^{(i)},y^{(i)},b) }_{ \text{average loss on trainign set} } + \underbrace{ \lambda R(\vec{w},b) }_{ \text{regularization} }
$$
- regularization is the bias towards a default answer

The key motivation for introducing loss functions is that not all errors should be treated the same. Points near the decision boundary are inherently uncertain and should be penalized less than those that are place far on the wrong side if the boundary.

# Large Margin Classifier 

$$
\gamma_{i}(\vec{w},b)= \frac{y^{(i)}(\vec{w}\cdot\vec{x}^{(i)}+b)}{\lvert \lvert \vec{w} \rvert  \rvert }
$$
- $\gamma_{i}>0$ is classified correctly 
- $\gamma_{i} <0$ is misclassified 

The absolute value of the margin is always the distance of the training point from the decision boundary. We define a $\gamma_{ref}$ which is the margin we want all training points to achieve. 
We can define the loss as a ratio of $\frac{\gamma_{i}}{\gamma_{ref}}$

$$
\text{Loss}_{h}\left( \frac{\gamma_{i}}{\gamma_{ref}} \right) = \begin{cases}
1-\frac{\gamma_{i}}{\gamma_{ref}}  & \text{if } \gamma<\gamma_{ref} \\
0  & \text{otherwise}
\end{cases}
$$

So the objective function becomes : 

$$
J(\vec{w},b,\gamma_{ref}) = \frac{1}{n}\sum_{i=1}^{n} \text{Loss}_{h}\left( \frac{\gamma_{i}}{\gamma_{ref}} \right) + \lambda\left( \frac{1}{\gamma_{ref}^{2}} \right)
$$

- So in this case we can optimize for getting the correct weights and obtaining a good margin
	- "Striking a balance between minimizing hinge loss and maximizing the margin"

- Can we find a expression for $\gamma_{ref}$?
	- Express it in units of $\lvert \lvert \vec{w} \rvert \rvert_{2}$
	- $\gamma_{ref}=\frac{1}{\lvert \lvert \vec{w} \rvert \rvert}_{2}$

Then the objective function becomes 

$$
J(\vec{w},b)=\frac{1}{n}\sum_{i=1}^{n} \text{Loss}_{h}(y^{(i)}(\vec{w}\cdot\vec{x}+b)) + \frac{\lambda}{2}\lvert \lvert \vec{w} \rvert  \rvert _{2}
$$
- for latter derivatives we use $\frac{\lambda}{2}$ . 
- the goal is to find $\vec{w},b$ that minimize $J(\vec{w},b)$. Now the choice of $\lvert \lvert \vec{w} \rvert \rvert$ matters a lot. 

# Gradient descent

Is an iterative optimization algorithm that is used to minimize a function by adjusting its parameters. The main idea behind the gradient descent is to move towards the minimum of a function by taking steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point
$$
\theta_{t+1} = \theta_{t}-\eta \nabla _{\theta}f(\theta_{t})
$$
- $\eta$ is the learning rate, a positive scalar determining the step size
- $\theta_{t}$ is the parameter vector at iteration $t$

Finding the correct learning rate is either done by experimentation or adaptive algorithms. 

## Stochastic Gradient descent (SDG) 

Uses **one training example** to compute the gradient for each optimization step. It is much faster, but **introduces more noise** into the optimization process, which can help escape local minima, but may also result in more fluctuations in the convergence path.


## Batch Gradient descent

Uses the entire data set to compute the gradient. Although it can converge to the global minimum for convex functions, it is **computationally expensive** for large datasets.

## Mini-Batch Gradient descent

Uses a **subset of the training data** (mini-batch) to compute the gradient. Thus, it balances computational efficiency and stability of the convergence process. **One usually refers to Mini-Batch GD when using the term SGD**, as the training set is shuffled anew after every epoch (full iteration over the entire training set) before renewed batch selection.


# Support Vector Machine (SWM)

The objective function of the support vector machine tries to 
	1) minimize the average Hinge loss on the training examples while 
	2) pushing the margin boundaries apart by reducing $||w||$. 
These two goals are in opposition to each other.

The parameter update : 
$$
\begin{align}
\vec{w_{t+1}}  & = \vec{w}-\eta \nabla_{\vec{w}}J(\vec{w},b) \\
 & =\vec{w} - \eta \nabla_{\vec{w}}\left[ \frac{\lambda}{2}\lvert \lvert \vec{w} \rvert  \rvert _{2}+\frac{1}{n}\sum_{i=1}^{n}\text{Loss}_{h}(y^{(i)}(\vec{w}\cdot\vec{x^{}}^{(i)}+b)) \right] \\
 & =\dots \\
 & =\vec{w}-\eta \lambda \vec{w} + \frac{\eta}{n}\sum_{i=1}^{n}\begin{cases}
y^{(i)}\vec{x}^{(i)}  & \text{if } y^{(i)}(\vec{w}\cdot\vec{x}+b) \leq 1 \\
0 & \text{otherwise}
\end{cases} \\
b_{t+1}  & =b +\frac{\eta}{n}\sum_{i=1}^{n}\begin{cases}
y^{(i)}  & \text{if } y^{(i)}(\vec{w}\cdot\vec{x}+b) \leq 1 \\
0 & \text{otherwise}
\end{cases}
\end{align}
$$

## The Algorithm 
![[Screenshot 2024-10-22 at 14.06.53.png]]

## The influence of $\lambda$

![[Screenshot 2024-10-22 at 14.10.04.png]]