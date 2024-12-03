# GridSearchCVPlugin

The `GridSearchCVPlugin` class is part of the `framework3.plugins.pipelines.grid` module and is designed to perform hyperparameter optimization on a `BaseFilter` using scikit-learn's `GridSearchCV`. This plugin automates the process of finding the best parameters for a given model by evaluating different parameter combinations through cross-validation.

## Module Contents
::: framework3.plugins.pipelines.grid.grid_pipeline
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

## Class Hierarchy
- GridSearchCVPlugin

## GridSearchCVPlugin
`GridSearchCVPlugin` extends `BasePlugin` and provides functionality for hyperparameter optimization in classification or regression models.

### Key Methods:
- `__init__(filterx: Type[BaseFilter], param_grid: Dict[str, Any], scoring: str, cv: int = 2)`: Initializes the plugin with the filter, parameter grid, evaluation metric, and number of folds for cross-validation.
- `fit(x: XYData, y: XYData) -> None`: Fits the `GridSearchCV` object to the provided data.
- `predict(x: XYData) -> XYData`: Makes predictions using the best estimator found by `GridSearchCV`.

## Usage Examples

### Creating and Using GridSearchCVPlugin
```python
from framework3.plugins.filters.clasification.svm import ClassifierSVMPlugin
from framework3.base.base_types import XYData
import numpy as np

# Create example data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])
X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
y_data = XYData(_hash='y_data', _path='/tmp', _value=y)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Create the GridSearchCVPlugin
grid_search = GridSearchCVPlugin(
    filterx=ClassifierSVMPlugin,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3
)

# Fit the grid search
grid_search.fit(X_data, y_data)

# Make predictions
X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
predictions = grid_search.predict(X_test)
print(predictions.value)

# Access the best parameters
print(grid_search._clf.best_params_)
```

## Best Practices
1. Ensure you correctly define the parameter grid for the model you are optimizing.
2. Use appropriate evaluation metrics for the problem you are solving (e.g., 'accuracy' for classification, 'neg_mean_squared_error' for regression).
3. Adjust the number of cross-validation folds (`cv`) according to the size of your dataset to avoid overfitting or underfitting.

## Conclusion
`GridSearchCVPlugin` provides an efficient way to optimize model hyperparameters within the `framework3` ecosystem. By following the best practices and examples provided, you can easily integrate this plugin into your machine learning workflows.
