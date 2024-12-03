from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    homogeneity_score,
    completeness_score,
)
from framework3.base.base_types import Float
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseMetric
from framework3.container.container import Container
from typing import Any

import numpy as np

__all__ = [
    "NMI",
    "ARI",
    "Silhouette",
    "CalinskiHarabasz",
    "Homogeneity",
    "Completeness",
]


@Container.bind()
class NMI(BaseMetric):
    """
    Normalized Mutual Information (NMI) metric for clustering evaluation.

    NMI is a normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation).

    Example:
    ```python
        >>> from framework3.plugins.metrics.clustering import NMI
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        >>> y_true = np.array([0, 0, 0, 1, 1, 1])
        >>> y_pred = np.array([0, 0, 1, 1, 1, 1])
        >>>
        >>> nmi_metric = NMI()
        >>> score = nmi_metric.evaluate(x_data, y_true, y_pred)
        >>> print(f"NMI Score: {score}")
    ```
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the Normalized Mutual Information score.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (Any): The ground truth labels.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Any): Additional keyword arguments passed to sklearn's normalized_mutual_info_score.

        Returns:
            (Float|np.ndarray): The NMI score.
        """
        return normalized_mutual_info_score(y_true, y_pred, **kwargs)


@Container.bind()
class ARI(BaseMetric):
    """
    Adjusted Rand Index (ARI) metric for clustering evaluation.

    The Adjusted Rand Index is the corrected-for-chance version of the Rand Index. It measures similarity between two clusterings, adjusted for chance.

    Example:
    ```python
        >>> from framework3.plugins.metrics.clustering import ARI
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        >>> y_true = np.array([0, 0, 0, 1, 1, 1])
        >>> y_pred = np.array([0, 0, 1, 1, 1, 1])
        >>>
        >>> ari_metric = ARI()
        >>> score = ari_metric.evaluate(x_data, y_true, y_pred)
        >>> print(f"ARI Score: {score}")
    ```
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the Adjusted Rand Index score.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (Any): The ground truth labels.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Any): Additional keyword arguments passed to sklearn's adjusted_rand_score.

        Returns:
            (Float|np.ndarray): The ARI score.
        """
        return adjusted_rand_score(y_true, y_pred, **kwargs)


@Container.bind()
class Silhouette(BaseMetric):
    """
    Silhouette Coefficient metric for clustering evaluation.

    The Silhouette Coefficient is calculated using the mean intra-cluster distance and the mean nearest-cluster distance for each sample.

    Example:
    ```python
        >>> from framework3.plugins.metrics.clustering import Silhouette
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        >>> y_pred = np.array([0, 0, 0, 1, 1, 1])
        >>>
        >>> silhouette_metric = Silhouette()
        >>> score = silhouette_metric.evaluate(x_data, None, y_pred)
        >>> print(f"Silhouette Score: {score}")
    ```
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the Silhouette Coefficient.

        Args:
            x_data (XYData): The input data.
            y_true (Any): Not used for this metric, but required by the interface.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Any): Additional keyword arguments passed to sklearn's silhouette_score.

        Returns:
            (Float|np.ndarray): The Silhouette Coefficient.
        """
        return silhouette_score(x_data.value, y_pred, **kwargs)


@Container.bind()
class CalinskiHarabasz(BaseMetric):
    """
    Calinski-Harabasz Index metric for clustering evaluation.

    The Calinski-Harabasz Index is the ratio of the sum of between-clusters dispersion and of inter-cluster dispersion for all clusters.

    Example:
    ```python
        >>> from framework3.plugins.metrics.clustering import CalinskiHarabasz
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        >>> y_pred = np.array([0, 0, 0, 1, 1, 1])
        >>>
        >>> ch_metric = CalinskiHarabasz()
        >>> score = ch_metric.evaluate(x_data, None, y_pred)
        >>> print(f"Calinski-Harabasz Score: {score}")
    ```
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the Calinski-Harabasz Index.

        Args:
            x_data (XYData): The input data.
            y_true (Any): Not used for this metric, but required by the interface.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Any): Additional keyword arguments passed to sklearn's calinski_harabasz_score.

        Returns:
            (Float|np.ndarray): The Calinski-Harabasz Index.
        """
        return calinski_harabasz_score(x_data.value, y_pred, **kwargs)


@Container.bind()
class Homogeneity(BaseMetric):
    """
    Homogeneity metric for clustering evaluation.

    Homogeneity measures whether all of its clusters contain only data points which are members of a single class.

    Example:
    ```python
        >>> from framework3.plugins.metrics.clustering import Homogeneity
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        >>> y_true = np.array([0, 0, 0, 1, 1, 1])
        >>> y_pred = np.array([0, 0, 1, 1, 1, 1])
        >>>
        >>> homogeneity_metric = Homogeneity()
        >>> score = homogeneity_metric.evaluate(x_data, y_true, y_pred)
        >>> print(f"Homogeneity Score: {score}")
    ```
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the Homogeneity score.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (Any): The ground truth labels.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Any): Additional keyword arguments passed to sklearn's homogeneity_score.

        Returns:
            (Float|np.ndarray): The Homogeneity score.
        """
        return homogeneity_score(y_true, y_pred, **kwargs)


@Container.bind()
class Completeness(BaseMetric):
    """
    Completeness metric for clustering evaluation.

    Completeness measures whether all members of a given class are assigned to the same cluster.

    Example:
    ```python
        >>> from framework3.plugins.metrics.clustering import Completeness
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        >>> y_true = np.array([0, 0, 0, 1, 1, 1])
        >>> y_pred = np.array([0, 0, 1, 1, 1, 1])
        >>>
        >>> completeness_metric = Completeness()
        >>> score = completeness_metric.evaluate(x_data, y_true, y_pred)
        >>> print(f"Completeness Score: {score}")
    ```
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the Completeness score.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (Any): The ground truth labels.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Any): Additional keyword arguments passed to sklearn's completeness_score.

        Returns:
            (Float|np.ndarray): The Completeness score.
        """
        return completeness_score(y_true, y_pred, **kwargs)
