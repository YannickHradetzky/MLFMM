Decision Trees are fundamental building blocks in many machine learning algorithms, including Random Forest and XGBoost. They are intuitive, easy to interpret, and powerful for both classification and regression tasks. A Decision Tree models a sequence of decisions and their possible consequences.

- **Root Node**: Top Node representing the entire dataset. (Starting Point)
- **Internal Node**: represent decisions based on feature values. Each internal Node split the data into subsets based on a specific feature ad a threshold criterion
- **Branches**: he connections between nodes, representing the outcome of a decision. Each branch corresponds to one of the possible values of the feature used at the internal node.
- **Leaf Nodes**: Terminal nodes that represent the final outcome, either a class label in classification or a continuous value in regression.

---
# Common Splitting Criteria
- **Gini Impurity**
- **Entropy (Information Gain)**
- **Variance Reduction** (for regression)

# Stopping Criteria
- Maximum tree depth
- Minimum number of samples required to split a node is not met
- All instances at a node belong to the same class (pure node)
- The reduction in impurity or variance is below a certain threshold

# Algorithm 

Initialize the root node with the entire dataset. Then, choose a feature and a threshold that best splits the data at the current node. Split the data into subsets based on the chosen feature and threshold. Create child nodes for each subset. For each child node, make the node a leaf, if the subset is pure (all instances have the same class or value) or a stopping criterion is met (e.g., maximum depth). Otherwise, keep repeating the process recursively for the subset at this child node

# Pros and Cons

## Pruning 
Pruning is a technique used to reduce the size of the tree by removing nodes that provide little power in classifying instances. This helps to combat overfitting and improves the tree’s generalization to new data

**Pros**
- Easy to visualize
- good for explanation to non experts
- capture non linear behavior
- no assumption regarding data distribution 
- it is easy to prune trees to combat overfitting
- **Easy access to feature importance**
	- A feature that contributed to a significant reduction in measure is important

**Cons**
- Easy to overfit
- Hard to compare (Nodes influence their children)
- bad when classes are very unbalanced

