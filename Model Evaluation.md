# Learning Curve and Batched Learning

For a Loss function like : 
$$
L = \frac{1}{n}\sum_{i=1}^{N}\text{Loss}(x^{(i)}, y^{(i)};\theta)
$$
See [[Linear Classification II - Objective Functions]]

**Learning Curves, loss (mean squared error) computed for current parameter values for the entire data set.** 

![[Screenshot 2024-11-05 at 13.08.03.png]]


**Left**: Gradient descent, each step includes computing the objective function and its gradient for the entire data set. The curve is extremely smooth. 

**Center**: Stochastic gradient descent that uses only a single sample per update step. Every step creates very different parameter updates, hence the loss function changes rapidly. Here only a fraction of the data set is used. 

**Right**: Batched stochastic gradient descent, with a batch size of 32. At every iteration a random 32 samples are used to compute the gradients of the objective function and update the parameters. At the end of the curve each data point has been used once

# Feature Normalization

Features often have drastically varying sizes and units. Therefore the gradient might come out funky since it has big changes in some and very small changes in other features (dimensions). Therefore we try to make all features have the same magnitude by normalization

$$
\tilde{\phi_{j}(\vec{x}^{(i)})} \leftarrow \frac{\phi_{j}(\vec{x}^{(i)})-\overbrace{ \bar{\phi_{j}}U }^{ \text{mean of feature j in training set} }}{\underbrace{ \sigma(\phi_{j}) }_{ \text{std of feature j in training} }}
$$

Then 
- $E[\phi_{j}]=0$
- $E[\phi_{j}^{2}]=1$

**Important** : Use the mean and std of the training set when normalizing the test Set

# Analyzing Classifier Performance
[[Soft Classification]]; [[Linear Classification]]

## Confusion Matrix

![[Pasted image 20241105133217.png]]

Confusion matrix that counts and divides up the samples by comparing ground truth and predicted label. The entries of the confusion matrix are used to determine the quality of the classifier

- False Positive are called Typ 1 Error
- False Negative are called Typ 2 Error

## Accuracy vs. Precision 

$$
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN} = 1-\text{missclassification error}
$$


$$
\text{Precision} = \frac{TP}{TP+FP} 
$$

Precision **is important** if the cost of a false positive is high. (e.g Toxicity, Cancer)

## Recall (True positive Rate)

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

## F1 Score
Is the harmonic mean of precision and recall, providing a single metric that balances both metrics.

$$
\text{F1}=2\cdot\frac{\text{Precission}\cdot \text{Recall}}{\text{Precission}+ \text{Recall}}
$$

## False positive Rate

$$
\text{FPR}= \frac{TN}{TN+FP}
$$
## Reciever Operating Characteristic (ROC)

![[Screenshot 2024-11-05 at 13.44.25.png]]

Determines how a probability is converted to a positive label. The area under the curve provides a single value to **determine the overall performance** of the classifier.

# Analyzing Regression Model Performance

[[MachineLearningForMoleculesAndMaterials.pdf]] (7.4)

