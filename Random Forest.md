Random Forest is an ensemble learning method that builds upon the strengths of Decision Trees while mitigating their weaknesses, particularly the tendency to overfit. It combines multiple Decision Trees to create a “forest” of trees, hence the name. **Each tree in the forest is trained on a different subset of the data**, and the final prediction is made by aggregating the predictions of all the trees. This method leverages the power of multiple models to improve accuracy and robustness.

Random Forest operates by **constructing a multitude of Decision Trees during training time** and outputting the **mode of the classes** (for classification) or the **mean prediction** (for regression) of the individual trees. It is a type of ensemble method known as bagging (Bootstrap Aggregating)

---
# Concepts

**Bagging**
By creating multiple subsets of the original data set through random sampling with replacement, each Decision Tree is trained on a slightly different data set. This introduces diversity among the trees, which helps reduce variance and prevent overfitting

**Feature Randomness**
In addition to bagging, Random Forest introduces an additional layer of randomness by selecting a random subset of features at each split in the tree. This ensures that the trees in the forest are less correlated with each other, further reducing variance.

**Out of Bag Error**
Since each tree in the Random Forest is trained on a bootstrap sample, approximately one-third of the data is not used in any given bootstrap sample. These unused data points are called **Out-of-Bag (OOB) samples**. The OOB samples can be used to estimate the model’s prediction error without the need for a separate validation set. The OOB error is calculated as the average error for each observation using only the trees that did not use that observation in their bootstrap sample