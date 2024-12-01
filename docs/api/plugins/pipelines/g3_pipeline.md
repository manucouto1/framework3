# GridSearchCVPipeline

::: framework3.plugins.pipelines.gs_cv_pipeline.GridSearchCVPipeline

## Overview

The GridSearchCVPipeline is an advanced pipeline implementation in the framework3 ecosystem that combines the power of scikit-learn's GridSearchCV with the flexibility of framework3's BaseFilters. It allows you to perform hyperparameter tuning across multiple steps in your machine learning pipeline.

## Key Features

- Integrates seamlessly with framework3's BaseFilters
- Performs grid search cross-validation on a sequence of filters
- Supports custom scoring metrics
- Allows for complex parameter grids across multiple pipeline steps

## Basic Usage

Here's a simple example of how to create and use a GridSearchCVPipeline:

```python
from framework3.plugins.pipelines import GridSearchCVPipeline
from framework3.plugins.filters.transformation import PCAPlugin
from framework3.plugins.filters.classification import ClassifierSVMPlugin
from framework3.base.base_types import XYData
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])
X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
y_data = XYData(_hash='y_data', _path='/tmp', _value=y)

# Define the parameter grid
param_grid = {
    'pca__n_components': [1, 2],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
}

# Create the GridSearchCVPipeline
grid_search = GridSearchCVPipeline(
    filterx=[
        PCAPlugin,
        ClassifierSVMPlugin,
    ],
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

## Advanced Usage

### Custom Scoring

You can use custom scoring functions or multiple scoring metrics:

```python
from sklearn.metrics import make_scorer, f1_score

# Custom scoring function
def custom_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

# Create GridSearchCVPipeline with custom scoring
grid_search = GridSearchCVPipeline(
    filterx=[PCAPlugin, ClassifierSVMPlugin],
    param_grid=param_grid,
    scoring={
        'accuracy': 'accuracy',
        'f1': make_scorer(custom_f1)
    },
    cv=3
)

# Fit and access results
grid_search.fit(X_data, y_data)
print(grid_search._clf.best_score_)
print(grid_search._clf.cv_results_)
```

### Complex Parameter Grids

You can define complex parameter grids that span multiple steps in your pipeline:

```python
from framework3.plugins.filters.feature_selection import VarianceThresholdPlugin

param_grid = {
    'pca__n_components': [2, 3, 4],
    'variancethreshold__threshold': [0.1, 0.2],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

grid_search = GridSearchCVPipeline(
    filterx=[
        PCAPlugin,
        VarianceThresholdPlugin,
        ClassifierSVMPlugin
    ],
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)
```

## API Reference

### GridSearchCVPipeline

```python
class GridSearchCVPipeline(BasePipeline):
    def __init__(self, filterx: List[type[BaseFilter]], param_grid: Dict[str, List[Any]], scoring: str|Callable|Tuple|Dict, cv: int = 2, metrics: List[BaseMetric] = []):
        """
        Initialize the GridSearchCVPipeline.

        Args:
            filterx (List[type[BaseFilter]]): List of BaseFilter classes defining the pipeline steps.
            param_grid (Dict[str, List[Any]]): Dictionary with parameters names as keys and lists of parameter settings to try as values.
            scoring (str|Callable|Tuple|Dict): Strategy to evaluate the performance of the cross-validated model on the test set.
            cv (int, optional): Determines the cross-validation splitting strategy. Defaults to 2.
            metrics (List[BaseMetric], optional): List of additional metrics to compute. Defaults to [].
        """
```

#### Methods

- `fit(self, x: XYData, y: Optional[XYData]) -> None`: Fit the GridSearchCV object to the given data.
- `predict(self, x: XYData) -> XYData`: Make predictions using the best estimator found by GridSearchCV.
- `evaluate(self, x_data: XYData, y_true: XYData|None, y_pred: XYData) -> Dict[str, float]`: Evaluate the performance of the best estimator.

## Best Practices

1. **Parameter Space**: Start with a broad parameter space and then narrow it down based on initial results.

2. **Computational Resources**: Be mindful of the computational cost when defining large parameter grids, especially with large datasets.

3. **Cross-Validation**: Choose an appropriate number of cross-validation splits (cv) based on your dataset size and computational resources.

4. **Scoring**: Use appropriate scoring metrics that align with your problem's objectives. Consider using multiple metrics to get a comprehensive view of model performance.

5. **Pipeline Design**: Carefully consider the order and types of filters in your pipeline to ensure they work well together and make sense for your data and problem.