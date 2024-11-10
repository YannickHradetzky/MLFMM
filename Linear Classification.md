#MLFMM 

For Notation see [[Notation and Math Intro]]
# Introduction (Chapter 1)
## Train a classifier
### Motivation 

Have a few labels and the algorithm does the rest. 
### Approach

- In binary classification  we have labels $y \in \{  -1, +1 \}$ 

- training Set $S_{n}=\{ (\underline{x^{(i)}},y^{(i)}) \}_{i=1}^{n}$

- classification maps from feature vectors to labels : $f:R^{d}\to \{ -1, +1 \}$

- we usually have a set of classifiers $\mathcal{H}$ and our goal is to choose $\hat{h}\in \mathcal{H}$ that has the best chance to classify unseen samples that were not in the training set.

- during training we have only $\mathcal{E}_{n}=\frac{1}{n}\sum_{i=1}^{n}\mathcal{1}[h(\underline{x^{(i)}}) \neq y^{(i)}]$

- There may be $\hat{h}\in \mathcal{H}$ with $\mathcal{E}_{n}=0$ tat are bad on new samples
	- does terrible on out of distribution 

- **test error** $\mathcal{E}_{n^{'}}=\frac{1}{n}\sum_{i=1}^{n^{'}}\mathcal{1}[\hat{h}(\underline{x^{(i)}})\neq y^{(i)}]$
- **training error** $\mathcal{E_{n}}=\frac{1}{n}\sum_{i=1}^{n}\underbrace{ \mathcal{1}[y^{(i)}(\vec{w}^{T}x^{(i)})+b \leq 0] }_{ \text{returns 1 if statement is true zero otherwise} }$
	- Include the zero because if we don't know then we are wrong

### important factors

- choose feature vectors

- number and distribution of training samples $S_{n}$

- selection of possibles classifiers $\mathcal{H}$

- how we select $\hat{h}\in \mathcal{H}$

