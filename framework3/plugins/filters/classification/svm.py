from typing import Any, Dict, List, Literal
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.container.container import Container
from sklearn.svm import SVC

L = Literal["linear", "poly", "rbf", "sigmoid"]
__all__ = ["ClassifierSVMPlugin"]


@Container.bind()
class ClassifierSVMPlugin(BaseFilter, BasePlugin):
    """
    A plugin for Support Vector Machine (SVM) classification using scikit-learn's SVC.

    This plugin wraps the SVC (Support Vector Classification) implementation from scikit-learn,
    providing an interface compatible with the framework3 ecosystem. It allows for easy
    integration of SVM classification into pipelines and supports grid search for hyperparameter tuning.

    Attributes:
        _model (SVC): The underlying scikit-learn SVC model.

    Args:
        C (float): Regularization parameter. Default is 1.0.
        gamma (float | Literal['scale', 'auto']): Kernel coefficient. Default is 'scale'.
        kernel (L): Specifies the kernel type to be used in the algorithm.
                    Can be 'linear', 'poly', 'rbf', or 'sigmoid'. Default is 'linear'.

    Example:
        ```python
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> # Create some sample data
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> y = np.array([0, 0, 1, 1])
        >>>
        >>> # Create and fit the SVM classifier
        >>> svm_plugin = ClassifierSVMPlugin(C=1.0, kernel='rbf', gamma='scale')
        >>> svm_plugin.fit(XYData(X), XYData(y))
        >>>
        >>> # Make predictions
        >>> new_data = np.array([[2.5, 3.5], [3.5, 4.5]])
        >>> predictions = svm_plugin.predict(XYData(new_data))
        >>> print(predictions.value)  # Output: [0 1] (example output)
        >>>
        >>> # Use item_grid for hyperparameter tuning
        >>> grid_params = ClassifierSVMPlugin.item_grid(C=[0.1, 1, 10], kernel=['linear', 'rbf'], gamma=['scale', 'auto'])
        >>> print(grid_params)
        ```
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: float | Literal["scale", "auto"] = "scale",
        kernel: L = "linear",
    ) -> None:
        super().__init__(C=C, kernel=kernel, gamma=gamma)
        self._model = SVC(C=C, kernel=kernel, gamma=gamma)

    def fit(self, x: XYData, y: XYData | None):
        """
        Fit the SVM model using the provided training data.

        Args:
            x (XYData): The input features for training.
            y (XYData | None): The target values for training. If None, the method does nothing.
        """
        if y is not None:
            self._model.fit(x.value, y.value)  # type: ignore

    def predict(self, x: XYData):
        """
        Make predictions using the fitted SVM model.

        Args:
            x (XYData): The input features for prediction.

        Returns:
            (XYData): The predicted labels wrapped in an XYData object.
        """
        return XYData.mock(self._model.predict(x.value))

    @staticmethod
    def item_grid(
        C: List[float],
        kernel: List[L],
        gamma: List[float] | List[Literal["scale", "auto"]] = ["scale"],  # type: ignore[assignment]
    ) -> Dict[str, List[Any]]:
        """
        Generate a parameter grid for hyperparameter tuning.

        This method is typically used in conjunction with grid search for finding optimal hyperparameters.

        Args:
            C (List[float]): List of regularization parameter values to try.
            kernel (List[L]): List of kernel types to try.
            gamma (List[float] | List[Literal['scale', 'auto']]): List of gamma values to try. Default is ['scale'].

        Returns:
            Dict[str, Any]: A dictionary containing the filter class and parameter grid for grid search.
        """
        return {
            "ClassifierSVMPlugin__C": C,
            "ClassifierSVMPlugin__kernel": kernel,
            "ClassifierSVMPlugin__gamma": gamma,
        }
