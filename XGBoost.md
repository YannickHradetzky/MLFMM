Extreme Gradient Boosting
Collection of many models just like [[Random Forest]]

# **Boosting**
Is an ensemble method. 
- Weak Learners: 
	- Simple models that strive to do just slightly better than a random model
		- So far all Models [[Linear Regression]] and [[Linear Classification]] are strong learners
		- e.G: Trees are weak learners. 
 - Gradient Boosting works via pseudo-residuals (grad of objective functions) [[Linear Classification II - Objective Functions]]
 

**Process**
Take a weak learner and check where the model went wrong. Build the next weak learner, where previous learned. Repeat this till stopping criterium is met. 

All weak learners together then form one strong/robust learner. 

# **Algorithm for XG-Boost** 
$$
\begin{align}
S_{n}  & = \{ \vec{x}^{(i)},y^{(i)} \}_{i=1}^{n} \\
T & = \text{Number of boosts} \\
\lambda & = \text{Learning Rate}
\end{align}
$$

1) Initialize model with constant predictions
$$
F_{0} = \text{argmin}_{y}\left( \sum_{i=1}^{n}L(y^{(i)},y) \right)
$$
2) For each boosting round do
	1) Compute the pseudo residuals
$$
r_{t}^{(i)}=-\left[ \frac{\partial L(y^{(i)},F_{t-1}(\vec{x}^{(i)}))}{\partial F_{t-1}(\vec{x}^{(i)})} \right]
$$
	2) Compute second-order gradients (Hessians) 
$$
h_{t}^{(i)}=-\left[ \frac{\partial^{2} L(y^{(i)},F_{t-1}(\vec{x}^{(i)}))}{(\partial F_{t-1}(\vec{x}^{(i)}))^{2}} \right]
$$

3) Fit a new slow learner $f_{t}$ to the residuals using the following objective function and the training set $S_{n}^{'}=\left\{  \vec{x}^{(i)},\frac{r_{t}^{(i)}}{h_{t}^{(i)}}  \right\}_{i=1}^{n}$

$$
L^{(t)}= \sum_{i=1}^{n} \frac{1}{2}h_{t}^{(i)}\left[ \frac{r_{t}^{(i)}}{h_{t}^{(i)}}-f_{t}(\vec{x}^{(i)}) \right] + \Omega(f_{t}) \approx \sum_{i=1}^{n}\left[ \frac{1}{2}h_{t}^{(i)}f_{t}(\vec{x}^{(i)})^{2}-r_{t}^{(i)}f_{t}(\vec{x}^{(i)}) \right] + \Omega (f_{t})
$$

	With $\Omega(f_{t})$ being the regularization term: 
	$$
\Omega(f_{t})=\alpha B+\frac{1}{2}\lambda \sum_{j=1}^{T}w_{j}^{2}
	$$
		Where $B$ is the number of leafs
		$w_{j}$ is the leave weight
		$\alpha$ and $\lambda$ are the regularization parameters
		
4) Update the Model by adding the new tree's predictions:
	$$
F_{t}= F_{t-1}+\eta f_{t}(\vec{x})
	$$
The complete model is then: 
$$
F(\vec{x}) = F_{0}+\sum_{t=1}^{T}\eta f_{t}(\vec{x})
$$
# Key Features

- Includes Regularization 
	- XGBoost includes L1 (Lasso) and L2 (Ridge) regularization in its objective function to control the complexity of the model. This helps prevent overfitting, especially when dealing with high- dimensional data.
- Shrinkage
	- The learning rate η controls the contribution of each tree to the final model. By setting a small learning rate, XGBoost gradually updates the model, allowing more trees to be added without overfitting. This is often referred to as shrinkage.
- Column Subsampling
	- XGBoost supports column subsampling, which means that for each tree (or even for each split within a tree), a random subset of features is selected. This adds diversity among the trees, reducing overfitting and improving generalization, similar to the Random Forest approach
- Handling Missing Data
	- XGBoost has a built-in capability to handle missing data. During the tree-building process, it automatically learns the best direction (left or right) for missing values, improving its robustness to incomplete datasets




