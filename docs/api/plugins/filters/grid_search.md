# Grid Search Filter

The Grid Search filter is a powerful tool for hyperparameter tuning in machine learning models. It systematically works through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance.

## Overview

Grid search is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid. It's an exhaustive search through a manually specified subset of the hyperparameter space of a learning algorithm.

## Parameters

- `estimator`: The base estimator to be tuned.
- `param_grid`: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
- `scoring`: Strategy to evaluate the performance of the cross-validated model on the test set.
- `cv`: Determines the cross-validation splitting strategy.
- `n_jobs`: Number of jobs to run in parallel.
- `verbose`: Controls the verbosity: the higher, the more messages.

## Usage

Here's an example of how to use the Grid Search filter:

```python
from framework3.plugins.filters.grid_search import GridSearchFilter
from framework3.base import XYData
from sklearn.svm import SVC

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# Create the Grid Search filter
grid_search = GridSearchFilter(
    estimator=SVC(),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Prepare your data
X = XYData(value=[[0, 0], [1, 1], [2, 2], [3, 3]])
y = XYData(value=[0, 1, 0, 1])

# Fit the Grid Search filter
grid_search.fit(X, y)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")

# Use the best estimator for predictions
X_new = XYData(value=[[4, 4], [5, 5]])
predictions = grid_search.predict(X_new)
print(f"Predictions: {predictions}")
```

## Best Practices

1. **Parameter Space**: Define a reasonable parameter space. Too large a space can be computationally expensive.
2. **Cross-Validation**: Use an appropriate cross-validation strategy for your data.
3. **Scoring**: Choose a scoring metric that aligns with your problem's goals.
4. **Computational Resources**: Be mindful of the computational resources required, especially for large datasets or complex models.

## Notes

- Grid Search can be computationally expensive, especially with large datasets or a wide parameter space.
- Consider using RandomizedSearchCV for a more efficient search when the parameter space is large.
- The best parameters found can be accessed via the `best_params_` attribute after fitting.
- The best estimator can be accessed via the `best_estimator_` attribute after fitting.

## See Also

- [Scikit-learn GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Hyperparameter Tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
