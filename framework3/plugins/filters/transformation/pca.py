from typing import Any, Dict, List, Optional
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter
from framework3.container.container import Container
from sklearn.decomposition import PCA

__all__ = ["PCAPlugin"]


@Container.bind()
class PCAPlugin(BaseFilter):
    """
    A plugin for performing Principal Component Analysis (PCA) on input data.

    This plugin uses scikit-learn's PCA implementation to reduce the dimensionality
    of the input data. It inherits from BaseFilter and is bound to the Container.

    Attributes:
        _pca (PCA): The scikit-learn PCA object used for dimensionality reduction.

    Args:
        n_components (int): The number of components to keep after dimensionality reduction.

    Examples:
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> # Create a PCAPlugin instance
        >>> pca_plugin = PCAPlugin(n_components=2)
        >>>
        >>> # Create some sample data
        >>> X = XYData.mock(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        >>> y = None  # PCA doesn't use y for fitting
        >>>
        >>> # Fit the PCA model
        >>> pca_plugin.fit(X, y)
        >>>
        >>> # Transform new data
        >>> new_data = XYData.mock(np.array([[2, 3, 4], [5, 6, 7]]))
        >>> transformed_data = pca_plugin.predict(new_data)
        >>> print(transformed_data.value)  # This will be a 2x2 array
    """

    def __init__(self, n_components: int):
        """
        Initialize the PCAPlugin.

        Args:
            n_components (int): The number of components to keep after dimensionality reduction.
        """
        super().__init__(
            n_components=n_components
        )  # Initialize the BaseFilter and BasePlugin parent classes.
        self._pca = PCA(n_components=n_components)

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        """
        Fit the PCA model to the input data.

        Args:
            x (XYData): The input features to fit the PCA model.
            y (Optional[XYData]): Not used in PCA, but required by the BaseFilter interface.

        Returns:
            None
        """
        self._pca.fit(x.value)

    def predict(self, x: XYData) -> XYData:
        """
        Apply dimensionality reduction to the input data.

        Args:
            x (XYData): The input features to transform.

        Returns:
            XYData: The transformed data with reduced dimensionality.
        """
        return XYData.mock(self._pca.transform(x.value))

    @staticmethod
    def item_grid(n_components: List[int]) -> Dict[str, Any]:
        return {"PCAPlugin__n_components": n_components}
