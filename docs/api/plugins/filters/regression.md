# Regression Filters

::: framework3.plugins.filters.regression

## Overview

The Regression Filters module in framework3 provides a collection of powerful regression algorithms that can be easily integrated into your machine learning pipelines. These filters are designed to work seamlessly with the framework3 ecosystem, providing a consistent interface and enhanced functionality for various regression tasks.

## Available Regression Algorithms

### Linear Regression

The Linear Regression algorithm is implemented in the `LinearRegressionFilter`. This fundamental regression method models the relationship between a dependent variable and one or more independent variables using a linear approach.

#### Usage

```python
from framework3.plugins.filters.regression.linear_regression import LinearRegressionFilter

linear_reg = LinearRegressionFilter(fit_intercept=True, normalize=False)
```

#### Parameters

- `fit_intercept` (bool): Whether to calculate the intercept for this model.
- `normalize` (bool): If True, the regressors X will be normalized before regression.

### Random Forest Regression

The Random Forest Regression algorithm is implemented in the `RandomForestRegressorFilter`. This ensemble learning method operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees.

#### Usage

```python
from framework3.plugins.filters.regression.random_forest import RandomForestRegressorFilter

rf_reg = RandomForestRegressorFilter(n_estimators=100, max_depth=None, min_samples_split=2)
```

#### Parameters

- `n_estimators` (int): The number of trees in the forest.
- `max_depth` (int or None): The maximum depth of the tree.
- `min_samples_split` (int): The minimum number of samples required to split an internal node.

## Comprehensive Example: Boston Housing Dataset Regression

In this example, we'll demonstrate how to use the Regression Filters with the Boston Housing dataset, showcasing both Linear Regression and Random Forest Regression, as well as integration with GridSearchCV for parameter tuning.

```python
from framework3.plugins.pipelines.gs_cv_pipeline import GridSearchCVPipeline
from framework3.plugins.filters.regression.linear_regression import LinearRegressionFilter
from framework3.plugins.filters.regression.random_forest import RandomForestRegressorFilter
from framework3.base.base_types import XYData
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XYData objects
X_train_data = XYData(_hash='X_train', _path='/tmp', _value=X_train)
y_train_data = XYData(_hash='y_train', _path='/tmp', _value=y_train)
X_test_data = XYData(_hash='X_test', _path='/tmp', _value=X_test)
y_test_data = XYData(_hash='y_test', _path='/tmp', _value=y_test)

# Linear Regression
linear_pipeline = GridSearchCVPipeline(
    filterx=[LinearRegressionFilter],
    param_grid=LinearRegressionFilter.item_grid(fit_intercept=[True, False], normalize=[True, False]),
    scoring='neg_mean_squared_error',
    cv=5
)

# Fit Linear Regression
linear_pipeline.fit(X_train_data, y_train_data)

# Make predictions
linear_predictions = linear_pipeline.predict(X_test_data)
print("Linear Regression Predictions:", linear_predictions.value)

# Random Forest Regression
rf_pipeline = GridSearchCVPipeline(
    filterx=[RandomForestRegressorFilter],
    param_grid=RandomForestRegressorFilter.item_grid(n_estimators=[50, 100, 200], max_depth=[None, 10, 20]),
    scoring='neg_mean_squared_error',
    cv=5
)

# Fit Random Forest Regression
rf_pipeline.fit(X_train_data, y_train_data)

# Make predictions
rf_predictions = rf_pipeline.predict(X_test_data)
print("Random Forest Predictions:", rf_predictions.value)

# Evaluate the models
linear_mse = mean_squared_error(y_test, linear_predictions.value)
linear_r2 = r2_score(y_test, linear_predictions.value)

rf_mse = mean_squared_error(y_test, rf_predictions.value)
rf_r2 = r2_score(y_test, rf_predictions.value)

print("Linear Regression MSE:", linear_mse)
print("Linear Regression R2 Score:", linear_r2)
print("Random Forest MSE:", rf_mse)
print("Random Forest R2 Score:", rf_r2)
```

This example demonstrates how to:

1. Load and prepare the Boston Housing dataset
2. Create XYData objects for use with framework3
3. Set up GridSearchCV pipelines for both Linear Regression and Random Forest Regression
4. Fit the models and make predictions
5. Evaluate the models using Mean Squared Error (MSE) and R2 Score

## Best Practices

1. **Data Preprocessing**: Ensure your data is properly preprocessed before applying regression filters. This may include scaling, normalization, handling missing values, and feature engineering.

2. **Feature Selection**: Use appropriate feature selection techniques to identify the most relevant features for your regression task.

3. **Algorithm Selection**: Choose the appropriate regression algorithm based on the characteristics of your data and the specific requirements of your problem.

4. **Parameter Tuning**: Use `GridSearchCVPipeline` to find the optimal parameters for your chosen regression algorithm, as demonstrated in the example.

5. **Model Evaluation**: Always evaluate your regression models using appropriate metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R2 Score.

6. **Cross-Validation**: Use cross-validation techniques to ensure your model generalizes well to unseen data.

7. **Regularization**: Consider using regularized regression techniques (e.g., Ridge, Lasso) to prevent overfitting, especially when dealing with high-dimensional data.

8. **Ensemble Methods**: Explore ensemble regression techniques, such as Random Forest or Gradient Boosting, to improve model performance and robustness.

## Conclusion

The Regression Filters module in framework3 provides a powerful set of tools for various regression tasks. By leveraging these filters in combination with other framework3 components, you can build efficient and effective regression pipelines. The example with the Boston Housing dataset demonstrates how easy it is to use these regression algorithms and integrate them with GridSearchCV for parameter tuning, all within the framework3 ecosystem.
