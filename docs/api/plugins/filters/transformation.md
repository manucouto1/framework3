# Transformation Filters

::: framework3.plugins.filters.transformation

## Overview

The Transformation Filters module in framework3 provides a collection of data transformation algorithms that can be easily integrated into your machine learning pipelines. These filters are designed to preprocess and transform your data, enhancing its quality and suitability for various machine learning tasks.

## Available Transformation Algorithms

### Standard Scaler

The Standard Scaler is implemented in the `StandardScalerFilter`. This transformation standardizes features by removing the mean and scaling to unit variance.

#### Usage

```python
from framework3.plugins.filters.transformation.standard_scaler import StandardScalerFilter

scaler = StandardScalerFilter(with_mean=True, with_std=True)
```

#### Parameters

- `with_mean` (bool): If True, center the data before scaling.
- `with_std` (bool): If True, scale the data to unit variance.

### Min-Max Scaler

The Min-Max Scaler is implemented in the `MinMaxScalerFilter`. This transformation scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

#### Usage

```python
from framework3.plugins.filters.transformation.minmax_scaler import MinMaxScalerFilter

scaler = MinMaxScalerFilter(feature_range=(0, 1))
```

#### Parameters

- `feature_range` (tuple): Desired range of transformed data.

### PCA (Principal Component Analysis)

PCA is implemented in the `PCAFilter`. This transformation performs linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

#### Usage

```python
from framework3.plugins.filters.transformation.pca import PCAFilter

pca = PCAFilter(n_components=2)
```

#### Parameters

- `n_components` (int or float): Number of components to keep. If float, it represents the proportion of variance to be retained.

## Comprehensive Example: Data Transformation Pipeline

In this example, we'll demonstrate how to use the Transformation Filters in a pipeline to preprocess data for a machine learning task.

```python
from framework3.plugins.pipelines.pipeline import Pipeline
from framework3.plugins.filters.transformation.standard_scaler import StandardScalerFilter
from framework3.plugins.filters.transformation.pca import PCAFilter
from framework3.plugins.filters.classification.svm import ClassifierSVMPlugin
from framework3.base.base_types import XYData
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XYData objects
X_train_data = XYData(_hash='X_train', _path='/tmp', _value=X_train)
y_train_data = XYData(_hash='y_train', _path='/tmp', _value=y_train)
X_test_data = XYData(_hash='X_test', _path='/tmp', _value=X_test)
y_test_data = XYData(_hash='y_test', _path='/tmp', _value=y_test)

# Create a pipeline with transformation filters and a classifier
pipeline = Pipeline([
    ('scaler', StandardScalerFilter()),
    ('pca', PCAFilter(n_components=2)),
    ('svm', ClassifierSVMPlugin(kernel='rbf'))
])

# Fit the pipeline
pipeline.fit(X_train_data, y_train_data)

# Make predictions
predictions = pipeline.predict(X_test_data)
print("Predictions:", predictions.value)

# Evaluate the model
accuracy = (predictions.value == y_test).mean()
print("Accuracy:", accuracy)
```

This example demonstrates how to:

1. Load and prepare the Iris dataset
2. Create XYData objects for use with framework3
3. Set up a pipeline that includes StandardScaler, PCA, and SVM classifier
4. Fit the pipeline and make predictions
5. Evaluate the model's accuracy

## Best Practices

1. **Order of Transformations**: Consider the order of your transformations carefully. For example, scaling should typically be done before PCA.

2. **Feature Selection**: Use PCA or other feature selection techniques to reduce dimensionality when appropriate.

3. **Scaling**: Always scale your features when using distance-based algorithms or algorithms that assume normally distributed data.

4. **Cross-Validation**: Use cross-validation to ensure your transformations generalize well to unseen data.

5. **Data Leakage**: Be cautious of data leakage. Fit your transformations only on the training data and then apply them to the test data.

6. **Interpretability**: Keep in mind that some transformations (like PCA) can make your features less interpretable.

7. **Domain Knowledge**: Use your domain knowledge to guide your choice of transformations.

8. **Outlier Handling**: Consider using robust scalers if your data contains outliers.

## Conclusion

The Transformation Filters module in framework3 provides essential tools for data preprocessing and feature engineering. By leveraging these filters in combination with other framework3 components, you can build efficient and effective machine learning pipelines. The example demonstrates how easy it is to use these transformation algorithms in a pipeline, preparing data for a classification task within the framework3 ecosystem.
