from framework3.base.base_clases import BaseFilter
from sklearn.base import BaseEstimator

from framework3.base.base_types import XYData
from framework3.base.exceptions import NotTrainableFilterError


class SkWrapper(BaseEstimator):
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

    def __init__(self, z_clazz: type[BaseFilter], **kwargs):
        """
        Initialize the SkFilterWrapper.

        Args:
            **kwargs: Keyword arguments to be passed to the wrapped BaseFilter class.
        """
        self._z_clazz: type[BaseFilter] = z_clazz
        self._model: BaseFilter
        self.kwargs = kwargs

    def get_zclazz(self) -> str:
        return self._z_clazz.__name__

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
        try:
            self._model.fit(XYData.mock(x), XYData.mock(y))
        except NotTrainableFilterError:
            self._model.init()

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

    def transform(self, x):
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
        return self.kwargs | {"z_clazz": self._z_clazz}

    def set_params(self, **parameters):
        """
        Set the parameters of the estimator.

        Args:
            **parameters: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        for param, value in parameters.items():
            if param == "z_clazz":
                self._z_clazz = value
            else:
                self.kwargs[param] = value
        self._model = self._z_clazz(**self.kwargs)
        return self
