from typing import Any, Dict, Type

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter
from framework3.container.container import Container
from sklearn.model_selection import GridSearchCV

from framework3.utils.skestimator import SkWrapper

__all__ = ["GridSearchCVPlugin"]


class SkFilterWrapper(BaseEstimator):
    """
    A wrapper class for BaseFilter that implements scikit-learn's BaseEstimator interface.

    This class allows BaseFilter objects to be used with scikit-learn's GridSearchCV.

    Attributes:
        z_clazz (Type[BaseFilter]): The BaseFilter class to be wrapped.

    Example:
        >>> from framework3.plugins.filters.clasification.svm import ClassifierSVMPlugin
        >>> import numpy as np
        >>>
        >>> # Create a sample BaseFilter
        >>> class SampleFilter(ClassifierSVMPlugin):
        ...     pass
        >>>
        >>> # Set the class to be wrapped
        >>> SkFilterWrapper.z_clazz = SampleFilter
        >>>
        >>> # Create an instance of SkFilterWrapper
        >>> wrapper = SkFilterWrapper(C=1.0, kernel='rbf')
        >>>
        >>> # Use the wrapper with sklearn's GridSearchCV
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> y = np.array([0, 0, 1, 1])
        >>> wrapper.fit(X, y)
        >>> print(wrapper.predict([[2.5, 3.5]]))
    """

    z_clazz: Type[BaseFilter]

    def __init__(self, clazz, **kwargs):
        """
        Initialize the SkFilterWrapper.

        Args:
            **kwargs: Keyword arguments to be passed to the wrapped BaseFilter class.
        """
        self._model = clazz(**kwargs)
        self.kwargs = kwargs

    def fit(self, x, y, *args, **kwargs):
        """
        Fit the wrapped model to the given data.

        Args:
            x: The input features.
            y: The target values.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            self: The fitted estimator.
        """
        self._model.fit(XYData.mock(x), XYData.mock(y))
        return self

    def predict(self, x):
        """
        Make predictions using the wrapped model.

        Args:
            x: The input features.

        Returns:
            The predicted values.
        """
        return self._model.predict(XYData.mock(x)).value

    def get_params(self, deep=True):
        """
        Get the parameters of the estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                         contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {**self.kwargs}

    def set_params(self, **parameters):
        """
        Set the parameters of the estimator.

        Args:
            **parameters: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        self._model = SkFilterWrapper.z_clazz(**parameters)
        return self


@Container.bind()
class GridSearchCVPlugin(BaseFilter):
    """
    GridSearchCVPlugin is a plugin designed to perform hyperparameter tuning on a BaseFilter
    using scikit-learn's GridSearchCV. It automates the process of finding the best parameters
    for a given model by evaluating different combinations of parameters through cross-validation.

    Attributes:
        _clf (GridSearchCV): The GridSearchCV object used for hyperparameter tuning.

    Args:
        filterx (Type[BaseFilter]): The BaseFilter class to be tuned. This is the model for which
                                    hyperparameters will be optimized.
        param_grid (Dict[str, Any]): A dictionary where keys are parameter names and values are
                                     lists of parameter settings to try. This defines the search space.
        scoring (str): The strategy to evaluate the performance of the cross-validated model on the test set.
                       Common options include 'accuracy', 'f1', 'roc_auc', etc.
        cv (int, optional): Determines the cross-validation splitting strategy. Defaults to 2. It specifies
                            the number of folds in a (Stratified)KFold.

    Methods:
        fit(x: XYData, y: XYData):
            Fits the GridSearchCV object to the provided data. It searches for the best parameter
            combination based on the provided scoring metric.

            Args:
                x (XYData): The input features wrapped in an XYData object.
                y (XYData): The target values wrapped in an XYData object.

        predict(x: XYData) -> XYData:
            Makes predictions using the best estimator found by GridSearchCV.

            Args:
                x (XYData): The input features wrapped in an XYData object.

            Returns:
                (XYData): The predicted values wrapped in an XYData object.

    Example:
        ```python
        >>> from framework3.plugins.filters.clasification.svm import ClassifierSVMPlugin
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> # Create sample data
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> y = np.array([0, 0, 1, 1])
        >>> X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
        >>> y_data = XYData(_hash='y_data', _path='/tmp', _value=y)
        >>>
        >>> # Define the parameter grid
        >>> param_grid = {
        ...     'C': [0.1, 1, 10],
        ...     'kernel': ['linear', 'rbf'],
        ...     'gamma': ['scale', 'auto']
        ... }
        >>>
        >>> # Create the GridSearchCVPlugin
        >>> grid_search = GridSearchCVPlugin(
        ...     filterx=ClassifierSVMPlugin,
        ...     param_grid=param_grid,
        ...     scoring='accuracy',
        ...     cv=3
        ... )
        >>>
        >>> # Fit the grid search
        >>> grid_search.fit(X_data, y_data)
        >>>
        >>> # Make predictions
        >>> X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
        >>> predictions = grid_search.predict(X_test)
        >>> print(predictions.value)
        >>>
        >>> # Access the best parameters
        >>> print(grid_search._clf.best_params_)
        ```
    """

    def __init__(
        self,
        filterx: Type[BaseFilter],
        param_grid: Dict[str, Any],
        scoring: str,
        cv: int = 2,
    ):
        """
        Initialize the GridSearchCVPlugin.

        Args:
            filterx (Type[BaseFilter]): The BaseFilter class to be tuned.
            param_grid (Dict[str, Any]): Dictionary with parameters names (string) as keys
                                         and lists of parameter settings to try as values.
            scoring (str): Strategy to evaluate the performance of the cross-validated model on the test set.
            cv (int, optional): Determines the cross-validation splitting strategy. Defaults to 2.
        """
        super().__init__(filterx=filterx, param_grid=param_grid, scoring=scoring, cv=cv)

        self._clf: GridSearchCV = GridSearchCV(
            estimator=Pipeline(steps=[(filterx.__name__, SkWrapper(filterx))]),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit=True,
            verbose=0,
        )

    def fit(self, x, y):
        """
        Fit the GridSearchCV object to the given data.

        Args:
            x (XYData): The input features.
            y (XYData): The target values.
        """
        self._clf.fit(x.value, y.value)  # type: ignore

    def predict(self, x):
        """
        Make predictions using the best estimator found by GridSearchCV.

        Args:
            x (XYData): The input features.

        Returns:
            XYData: The predicted values wrapped in an XYData object.
        """
        return XYData.mock(self._clf.predict(x.value))  # type: ignore
