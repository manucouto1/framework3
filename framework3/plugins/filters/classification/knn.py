from framework3.base.base_clases import BaseFilter
from framework3.container.container import Container
from framework3.base.base_types import XYData
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Literal, Optional, Dict, Any

__all__ = ["KnnFilter"]


@Container.bind()
class KnnFilter(BaseFilter):
    """
    A wrapper for scikit-learn's KNeighborsClassifier using the framework3 BaseFilter interface.

    This filter implements the K-Nearest Neighbors algorithm for classification.

    Attributes:
        _clf (KNeighborsClassifier): The underlying scikit-learn KNN classifier.

    Example:
        ```python

        >>> from framework3.plugins.filters.classification.knn import KnnFilter
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> # Create sample data
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> y = np.array([0, 0, 1, 1])
        >>> X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
        >>> y_data = XYData(_hash='y_data', _path='/tmp', _value=y)
        >>>
        >>> # Create and fit the KNN filter
        >>> knn = KnnFilter(n_neighbors=3, weights='uniform')
        >>> knn.fit(X_data, y_data)
        >>>
        >>> # Make predictions
        >>> X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
        >>> predictions = knn.predict(X_test)
        >>> print(predictions.value)
        ```
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        metric_params: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize the KnnFilter.

        Args:
            n_neighbors (int): Number of neighbors to use. Defaults to 5.
            weights (str): Weight function used in prediction. Possible values:
                'uniform', 'distance'. Defaults to 'uniform'.
            algorithm (str): Algorithm used to compute the nearest neighbors.
                Options are 'ball_tree', 'kd_tree', 'brute', 'auto'. Defaults to 'auto'.
            leaf_size (int): Leaf size passed to BallTree or KDTree. Defaults to 30.
            p (int): Power parameter for the Minkowski metric. Defaults to 2 (Euclidean distance).
            metric (str): The distance metric to use. Defaults to 'minkowski'.
            metric_params (dict, optional): Additional keyword arguments for the metric function.
            n_jobs (int, optional): The number of parallel jobs to run for neighbors search.
        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self._clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        """
        Fit the KNN model to the given data.

        Args:
            x (XYData): The input features.
            y (XYData): The target values.
        """
        self._clf.fit(x.value, y.value)  # type: ignore

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted KNN model.

        Args:
            x (XYData): The input features to predict.

        Returns:
            XYData: The predicted values.
        """
        predictions = self._clf.predict(x.value)
        return XYData.mock(predictions)

    @staticmethod
    def item_grid(**kwargs) -> tuple[type[BaseFilter], Dict[str, List[Any]]]:
        """
        Generate a parameter grid for hyperparameter tuning.

        Args:
            **kwargs (Any): Keyword arguments to override default parameter ranges.

        Returns:
            Dict[str, Any]: A dictionary of parameter names and their possible values.
        """

        return KnnFilter, kwargs
