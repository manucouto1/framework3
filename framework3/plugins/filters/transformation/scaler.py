from typing import Optional
from sklearn.preprocessing import StandardScaler
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter
from framework3.container.container import Container


__all__ = ["StandardScalerPlugin"]


@Container.bind()
class StandardScalerPlugin(BaseFilter):
    """
    A plugin for standardizing features by removing the mean and scaling to unit variance.

    This plugin uses scikit-learn's StandardScaler to standardize features. It inherits from
    both BaseFilter and BasePlugin, and is bound to the Container.

    Attributes:
        scaler (StandardScaler): The scikit-learn StandardScaler object used for standardization.

    Examples:
        >>> import numpy as np
        >>> from framework3.base.base_types import XYData
        >>>
        >>> # Create a StandardScalerPlugin instance
        >>> scaler_plugin = StandardScalerPlugin()
        >>>
        >>> # Create some sample data
        >>> X = XYData.mock(np.array([[0, 0], [0, 0], [1, 1], [1, 1]]))
        >>> y = None  # StandardScaler doesn't use y for fitting
        >>>
        >>> # Fit the StandardScaler
        >>> scaler_plugin.fit(X, y)
        >>>
        >>> # Transform new data
        >>> new_data = XYData.mock(np.array([[2, 2], [-1, -1]]))
        >>> scaled_data = scaler_plugin.predict(new_data)
        >>> print(scaled_data.value)
        >>> # Output will be standardized, with mean 0 and unit variance
        >>> # For example: [[ 1.41421356  1.41421356]
        >>> #               [-1.41421356 -1.41421356]]
    """

    def __init__(self):
        """
        Initialize the StandardScalerPlugin.
        """
        super().__init__()  # Call the BaseFilter constructor to initialize the plugin's parameters
        self._scaler = StandardScaler()

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        """
        Fit the StandardScaler to the input data.

        Compute the mean and std to be used for later scaling.

        Args:
            x (XYData): The input features to fit the StandardScaler.
            y (Optional[XYData]): Not used in StandardScaler, but required by the BaseFilter interface.

        Returns:
            None
        """
        self._scaler.fit(x.value)

    def predict(self, x: XYData) -> XYData:
        """
        Perform standardization by centering and scaling the input data.

        Args:
            x (XYData): The input features to standardize.

        Returns:
            XYData: The standardized version of the input data.
        """
        return XYData.mock(self._scaler.transform(x.value))
