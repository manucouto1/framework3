from typing import Any, Callable, Dict, List, Tuple, Optional

from sklearn.model_selection import GridSearchCV
from framework3.base import BaseMetric, BaseFilter, XYData
from framework3.base.base_optimizer import BaseOptimizer
from framework3.container.container import Container
from sklearn.pipeline import Pipeline

from framework3.utils.skestimator import SkWrapper
from rich import print

__all__ = ["SklearnOptimizer"]


@Container.bind()
class SklearnOptimizer(BaseOptimizer):
    def __init__(
        self,
        scoring: str | Callable | Tuple | Dict,
        pipeline: BaseFilter | None = None,
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
            scoring=scoring,
            cv=cv,
            metrics=metrics,
            pipeline=pipeline,
        )

        self._grid = {}

    def get_grid(self, aux: Dict[str, Any]) -> None:
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    self.get_grid(filter_config)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                self.get_grid(pipeline)
            case _:
                if "_grid" in aux:
                    for param, value in aux["_grid"].items():
                        if type(value) is list:
                            self._grid[f'{aux["clazz"]}__{param}'] = value
                        else:
                            self._grid[f'{aux["clazz"]}__{param}'] = [value]

    def optimize(self, pipeline: BaseFilter):
        """Initialize the pipeline (e.g., set up logging)."""
        self.pipeline = pipeline
        self._filters = list(
            map(lambda x: (x.__name__, SkWrapper(x)), self.pipeline.get_types())
        )

        dumped_pipeline = self.pipeline.item_dump(include=["_grid"])
        print(dumped_pipeline)
        self.get_grid(dumped_pipeline)

        # for filter_config in dumped_pipeline["params"]["filters"]:
        #     if "_grid" in filter_config:
        #         filter_config["params"].update(**filter_config["_grid"])
        #         for k, v in filter_config["params"].items():
        #             if type(v) is list:
        #                 param_grid[f'{filter_config["clazz"]}__{k}'] = v
        #             else:
        #                 param_grid[f'{filter_config["clazz"]}__{k}'] = [v]

        self._pipeline = Pipeline(self._filters)

        self._clf: GridSearchCV = GridSearchCV(
            estimator=self._pipeline,
            param_grid=self._grid,
            scoring=self.scoring,
            cv=self.cv,
            verbose=1,
        )

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

    def fit(self, x: XYData, y: Optional[XYData]) -> None | float:
        """
        Fit the GridSearchCV object to the given data.

        Args:
            x (XYData): The input features.
            y (Optional[XYData]): The target values.
        """
        self._clf.fit(x.value, y.value if y is not None else None)
        return self._clf.best_score_  # type: ignore

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
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline using the provided metrics.

        This method applies each metric in the pipeline to the predicted and true values,
        returning a dictionary of results.

        Args:
            x_data (XYData): Input data.
            y_true (XYData|None): True target data.
            y_pred (XYData): Predicted target data.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results for each metric.

        Example:
            >>> evaluation = pipeline.evaluate(x_test, y_test, predictions)
            >>> print(evaluation)
            {'F1Score': 0.85}
        """
        results = {}
        for metric in self.metrics:
            results[metric.__class__.__name__] = metric.evaluate(x_data, y_true, y_pred)
        results["best_score"] = self._clf.best_score_  # type: ignore
        return results

    def log_metrics(self):
        """Log metrics (to be implemented)."""
        # TODO: Implement metric logging

    def finish(self):
        """Finish pipeline execution (e.g., close logger)."""
        # TODO: Finalize logger, possibly wandb
