from typing import Any, Callable, Dict, List, Tuple, Optional

from sklearn.model_selection import GridSearchCV
from framework3.base import XYData
from framework3.base import BaseFilter, BaseMetric, GridPipeline
from framework3.container.container import Container
from sklearn.pipeline import Pipeline

from framework3.utils.skestimator import SkWrapper


__all__ = ["GridSearchPipeline"]


@Container.bind()
class GridSearchPipeline(GridPipeline):
    """
    A pipeline that performs grid search cross-validation on a sequence of BaseFilters.

    This pipeline uses scikit-learn's GridSearchCV to perform hyperparameter tuning
    on a given sequence of BaseFilters.

    Attributes:
        _pipeline (F3Pipeline): The internal pipeline of filters.
        _clf (GridSearchCV): The GridSearchCV object used for hyperparameter tuning.

    Example:
        ```python
        from framework3.plugins.filters.clasification.svm import ClassifierSVMPlugin
        from framework3.plugins.filters.transformation.pca import PCAPlugin
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
        y_data = XYData(_hash='y_data', _path='/tmp', _value=y)

        # Define the parameter grid
        param_grid = {
            'pca__n_components': [1, 2],
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf'],
        }

        # Create the GridSearchCVPipeline
        grid_search = GridSearchCVPipeline(
            filters=[
                ('pca', PCAPlugin()),
                ('svm', ClassifierSVMPlugin()),
            ],
            param_grid=param_grid,
            scoring='accuracy',
            cv=3
        )

        # Fit the grid search
        grid_search.fit(X_data, y_data)

        # Make predictions
        X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
        predictions = grid_search.predict(X_test)
        print(predictions.value)

        # Access the best parameters
        print(grid_search._clf.best_params_)
        ```
    """

    def __init__(
        self,
        filterx: List[type[BaseFilter]],
        param_grid: Dict[str, List[Any]],
        scoring: str | Callable | Tuple | Dict,
        cv: int = 2,
        metrics: List[BaseMetric] = [],
    ):
        """
        Initialize the GridSearchCVPipeline.

        Args:
            filterx (List[Tuple[str, BaseFilter]]): List of (name, filter) tuples defining the pipeline steps.
            param_grid (Dict[str, Any]): Dictionary with parameters names (string) as keys
                                         and lists of parameter settings to try as values.
            scoring (str): Strategy to evaluate the performance of the cross-validated model on the test set.
            cv (int, optional): Determines the cross-validation splitting strategy. Defaults to 2.
        """
        super().__init__(
            filters=filterx,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            metrics=metrics,
        )

        self._filters = list(map(lambda x: (x.__name__, SkWrapper(x)), filterx))

        self._param_grid = param_grid

        self._pipeline = Pipeline(self._filters)

        self._clf: GridSearchCV = GridSearchCV(
            estimator=self._pipeline,
            param_grid=self._param_grid,
            scoring=scoring,
            cv=cv,
            verbose=0,
        )

    def init(self):
        """Initialize the pipeline (e.g., set up logging)."""
        # TODO: Initialize logger, possibly wandb

    def start(
        self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
    ) -> Optional[XYData]:
        """
        Start the pipeline execution.

        Args:
            x (XYData): Input data for fitting.
            y (Optional[XYData]): Target data for fitting.
            X_ (Optional[XYData]): Data for prediction (if different from x).

        Returns:
            Optional[XYData]: Prediction results if X_ is provided, else None.

        Raises:
            Exception: If an error occurs during pipeline execution.
        """
        try:
            self.fit(x, y)
            if X_ is not None:
                return self.predict(X_)
            else:
                return self.predict(x)
        except Exception as e:
            print(f"Error during pipeline execution: {e}")
            raise e

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        """
        Fit the GridSearchCV object to the given data.

        Args:
            x (XYData): The input features.
            y (Optional[XYData]): The target values.
        """
        self._clf.fit(x.value, y.value if y is not None else None)

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the best estimator found by GridSearchCV.

        Args:
            x (XYData): The input features.

        Returns:
            XYData: The predicted values wrapped in an XYData object.
        """
        return XYData.mock(self._clf.predict(x.value))  # type: ignore

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, float]:
        """
        Evaluate the performance of the best estimator.

        Args:
            x_data (XYData): The input features.
            y_true (XYData): The true target values.
            y_pred (XYData): The predicted target values.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        return {"best_score": self._clf.best_score_}

    def log_metrics(self):
        """Log metrics (to be implemented)."""
        # TODO: Implement metric logging

    def finish(self):
        """Finish pipeline execution (e.g., close logger)."""
        # TODO: Finalize logger, possibly wandb
