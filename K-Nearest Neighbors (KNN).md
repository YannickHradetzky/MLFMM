
KNN works by identifying the k training samples closest in distance to a new input, and then predicting the output based on the values of these neighbors. It is particularly effective for non-linear problems as it does not assume any underlying data distribution, hence non-parametric.

# Algorithm 

## Classification
```python 
# Input: 
# dataset: A list of tuples [(x1, y1), (x2, y2), ...] where xi is the feature vector and yi is the label
# query_point: The point for which we want to predict the label
# k: The number of neighbors to consider
def knn_classification(dataset, query_point, k):
    # Step 1: Calculate distances from query_point to all points in the dataset
    distances = []
    for data_point, label in dataset:
        distance = compute_distance(query_point, data_point)
        distances.append((distance, label))
    # Step 2: Sort the distances by ascending order
    distances.sort(key=lambda x: x[0])
    # Step 3: Select the k nearest neighbors
    k_neighbors = distances[:k]
    # Step 4: Count the labels of the k neighbors
    label_counts = {}
    for _, label in k_neighbors:
        label_counts[label] = label_counts.get(label, 0) + 1
    # Step 5: Determine the most frequent label
    predicted_label = max(label_counts, key=label_counts.get)
    return predicted_label
```
## Regression
```python
# Input:
# dataset: A list of tuples [(x1, y1), (x2, y2), ...] where xi is the feature vector and yi is the target value
# query_point: The point for which we want to predict the target value
# k: The number of neighbors to consider
def knn_regression(dataset, query_point, k):
    # Step 1: Calculate distances from query_point to all points in the dataset
    distances = []
    for data_point, target in dataset:
        distance = compute_distance(query_point, data_point)
        distances.append((distance, target))
    # Step 2: Sort the distances by ascending order
    distances.sort(key=lambda x: x[0])
    # Step 3: Select the k nearest neighbors
    k_neighbors = distances[:k]
    # Step 4: Compute the average of the target values of the k neighbors
    target_sum = sum(target for _, target in k_neighbors)
    predicted_value = target_sum / k
    return predicted_value
```

