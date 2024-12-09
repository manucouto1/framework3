from framework3.base import BaseFilter, BasePlugin, XYData
from framework3.container.container import Container
from sklearn.linear_model import LogisticRegression

__all__ = ["LogistiRegressionlugin"]

Container.bind()


class LogistiRegressionlugin(BaseFilter, BasePlugin):
    """
    A plugin that implements logistic regression using scikit-learn's LogisticRegression.

    This plugin wraps the LogisticRegression model from scikit-learn and adapts it
    to work within the framework3 ecosystem.

    Attributes:
        _logistic (LogisticRegression): The underlying scikit-learn LogisticRegression model.

    Example:
    ```python
        >>> import numpy as np
        >>> from framework3.base import XYData
        >>>
        >>> # Create sample data
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> y = np.array([0, 0, 1, 1])
        >>> X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
        >>> y_data = XYData(_hash='y_data', _path='/tmp', _value=y)
        >>>
        >>> # Create and fit the LogistiRegressionlugin
        >>> log_reg = LogistiRegressionlugin(max_ite=100, tol=1e-4)
        >>> log_reg.fit(X_data, y_data)
        >>>
        >>> # Make predictions
        >>> X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
        >>> predictions = log_reg.predict(X_test)
        >>> print(predictions.value)
        >>>
        >>> # Access the underlying scikit-learn model
        >>> print(log_reg._logistic.coef_)
    ```
    """

    def __init__(self, max_ite: int, tol: float):
        """
        Initialize the LogistiRegressionlugin.

        Args:
            max_ite (int): Maximum number of iterations for the solver to converge.
            tol (float): Tolerance for stopping criteria.
        """
        self._logistic = LogisticRegression(max_iter=max_ite, tol=tol)

    def fit(self, x: XYData, y: XYData | None) -> None:
        """
        Fit the logistic regression model.

        Args:
            x (XYData): The input features.
            y (XYData | None): The target values.

        Raises:
            ValueError: If y is None.

        Example:
        ```python
            >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
            >>> y = np.array([0, 0, 1, 1])
            >>> X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
            >>> y_data = XYData(_hash='y_data', _path='/tmp', _value=y)
            >>> log_reg = LogistiRegressionlugin(max_ite=100, tol=1e-4)
            >>> log_reg.fit(X_data, y_data)
        ```
        """
        if y is None:
            raise ValueError(
                "Target values (y) cannot be None for logistic regression."
            )
        self._logistic.fit(x._value, y._value)  # type: ignore

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted logistic regression model.

        Args:
            x (XYData): The input features.

        Returns:
            XYData: The predicted values wrapped in an XYData object.

        Example:
        ```python
            >>> X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
            >>> predictions = log_reg.predict(X_test)
            >>> print(predictions.value)
        ```
        """
        return XYData.mock(self._logistic.predict(x.value))
