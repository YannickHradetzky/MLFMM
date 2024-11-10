#MLFMM 

Hard classifiers only return labels for classes. See [[Linear Classification]] and [[Linear Classification II - Objective Functions]]. They need special training algorithms which often don't allow Statistical Gradient Descent. 

having a soft, differentiable, classification allows us to use regression ([[Linear Regression]]) in classification
# Softmax Classifier / Logistic Regression

So we are gonna use the sigmoid function on our prediction : 

$$
h(x) = \sigma(\vec{w}\vec{x}+b)
$$

with $\sigma(z) = \frac{1}{1+e^{-z}}$

- The inputs can be in the range of $[-\infty,\infty]$ and are called **log-odds** or **logits**
	- odds are ratios of probabilities $\frac{p}{1-p}$


## Loss Function 

$$
L(\theta) = -\sum_{i=1}^{K} y_{i}\ln \hat{y_{i}}
$$

The non-regularized objective function of a binary classifier would be : 

$$
J_{n}(\theta)=-\frac{1}{n}\sum_{i=1}^{n}[y^{(i)}\ln \sigma(\vec{w}\vec{x}^{(i)}+b)+(1-y^{(i)})\ln(1-\sigma(\vec{w}\vec{x}^{(i)}+b))]
$$

However, to avoid numerical instability in practice when using cross-entropy sigmoid classification you should use a model that outputs the logits and you use a loss function that works on logits instead of probability, e.g

$$
L(\vec{x}^{(i)},\theta,y^{(i)}) = max(\vec{w}\vec{x}^{(i)}+b,0)-(\vec{w}\vec{x}^{(i)}+b)y^{(i)}+\ln(1+e^{-\lvert \vec{w}\vec{x}^{(i)}+b \rvert })
$$

