from framework3.base.base_clases import BaseFilter
from framework3.container.container import Container
from framework3.base.base_types import XYData
from sklearn.cluster import KMeans
from typing import Literal, Optional, Dict, Any

__all__ = ["KMeansFilter"]


@Container.bind()
class KMeansFilter(BaseFilter):
    """
    A wrapper for scikit-learn's KMeans clustering algorithm using the framework3 BaseFilter interface.

    This filter implements the K-Means clustering algorithm.

    Attributes:
        _clf (KMeans): The underlying scikit-learn KMeans clustering model.

    Example:
        ```python

        >>> from framework3.plugins.filters.clustering.kmeans import KMeansFilter
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> # Create sample data
        >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
        >>>
        >>> # Create and fit the KMeans filter
        >>> kmeans = KMeansFilter(n_clusters=2, random_state=42)
        >>> kmeans.fit(X_data)
        >>>
        >>> # Make predictions
        >>> X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[0, 0], [4, 4]]))
        >>> predictions = kmeans.predict(X_test)
        >>> print(predictions.value)
        ```
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        algorithm: Literal["lloyd", "elkan"] = "lloyd",
    ):
        """
        Initialize the KMeansFilter.

        Args:
            n_clusters (int): The number of clusters to form. Defaults to 8.
            init (str): Method for initialization. Defaults to 'k-means++'.
            n_init (int): Number of time the k-means algorithm will be run with different
                centroid seeds. Defaults to 10.
            max_iter (int): Maximum number of iterations of the k-means algorithm for a
                single run. Defaults to 300.
            tol (float): Relative tolerance with regards to Frobenius norm of the difference
                in the cluster centers of two consecutive iterations to declare convergence.
                Defaults to 1e-4.
            random_state (int, optional): Determines random number generation for centroid
                initialization.
            algorithm (str): K-means algorithm to use. Possible values: 'auto', 'full', 'elkan'.
                Defaults to 'auto'.
        """
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            algorithm=algorithm,
        )
        self._clf = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            algorithm=algorithm,
        )

    def fit(self, x: XYData, y: Optional[XYData] = None) -> None:
        """
        Fit the KMeans model to the given data.

        Args:
            x (XYData): The input features.
            y (XYData, optional): Not used, present for API consistency.
        """
        self._clf.fit(x.value)

    def predict(self, x: XYData) -> XYData:
        """
        Predict the closest cluster for each sample in X.

        Args:
            x (XYData): The input features to predict.

        Returns:
            XYData: The predicted cluster labels.
        """
        predictions = self._clf.predict(x.value)
        return XYData.mock(predictions)

    def transform(self, x: XYData) -> XYData:
        """
        Transform X to a cluster-distance space.

        Args:
            x (XYData): The input features to transform.

        Returns:
            XYData: The transformed data.
        """
        transformed = self._clf.transform(x.value)
        return XYData.mock(transformed)

    @staticmethod
    def item_grid(**kwargs) -> Dict[str, Any]:
        """
        Generate a parameter grid for hyperparameter tuning.

        Args:
            **kwargs (Any): Keyword arguments to override default parameter ranges.

        Returns:
            Dict[str, Any]: A dictionary of parameter names and their possible values.
        """

        return dict(map(lambda x: (f"KMeansFilter__{x[0]}", x[1]), kwargs.items()))
