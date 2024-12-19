from copy import deepcopy
from typing import List, Dict, Any, Sequence
from framework3.base import XYData
from framework3.base import BaseFilter
from framework3.base import ParallelPipeline
from framework3.base.exceptions import NotTrainableFilterError
from framework3.container.container import Container

__all__ = ["MonoPipeline"]


@Container.bind()
class MonoPipeline(ParallelPipeline):
    """
    A pipeline that combines multiple filters in parallel and constructs new features from their outputs.

    This class allows you to run multiple filters simultaneously on the same input data,
    and then combine their outputs to create new features. It's particularly useful for
    feature engineering and ensemble methods.

    Attributes:
        filters (Sequence[BaseFilter]): List of filters to be run in parallel.

    Example:
        >>> from framework3.plugins.filters.transformation import PCAPlugin
        >>> from framework3.plugins.filters.classification import KnnFilter
        >>> from framework3.plugins.metrics import F1Score
        >>>
        >>> pipeline = MonoPipeline(
        ...     filters=[
        ...         PCAPlugin(n_components=2),
        ...         KnnFilter(n_neighbors=3)
        ...     ],
        ...     metrics=[F1Score()]
        ... )
        >>>
        >>> # Assuming x_train, y_train, x_test, y_test are your data
        >>> pipeline.fit(x_train, y_train)
        >>> predictions = pipeline.predict(x_test)
        >>> evaluation = pipeline.evaluate(x_test, y_test, predictions)
        >>> print(evaluation)
    """

    def __init__(self, filters: Sequence[BaseFilter]):
        """
        Initialize the MonoPipeline.

        Args:
            filters (Sequence[BaseFilter]): List of filters to be run in parallel.
        """
        super().__init__(filters=filters)
        self.filters = filters

    def fit(self, x: XYData, y: XYData | None = None):
        """
        Fit all filters in parallel.

        This method applies the fit operation to each filter in the pipeline
        using the provided input data.

        Args:
            x (XYData): Input data.
            y (XYData, optional): Target data. Defaults to None.

        Example:
            >>> pipeline.fit(x_train, y_train)
        """
        for filter in self.filters:
            try:
                filter.fit(deepcopy(x), y)
            except NotTrainableFilterError:
                filter.init()

    def predict(self, x: XYData) -> XYData:
        """
        Run predictions on all filters in parallel and combine their outputs.

        This method applies the predict operation to each filter in the pipeline
        and then combines the outputs using the combine_features method.

        Args:
            x (XYData): Input data.

        Returns:
            XYData: Combined output from all filters.

        Example:
            >>> predictions = pipeline.predict(x_test)
        """
        outputs: List[XYData] = [filter.predict(deepcopy(x)) for filter in self.filters]
        return self.combine_features(outputs)

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
        return results

    @staticmethod
    def combine_features(pipeline_outputs: list[XYData]) -> XYData:
        """
        Combine features from all filter outputs.

        This method concatenates the features from all filter outputs along the last axis.

        Args:
            pipeline_outputs (List[XYData]): List of outputs from each filter.

        Returns:
            XYData: Combined output with concatenated features.

        Note:
            This method assumes that all filter outputs can be concatenated along the last axis.
            Make sure that your filters produce compatible outputs.
        """
        return XYData.concat(
            [XYData.ensure_dim(output.value) for output in pipeline_outputs], axis=-1
        )

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
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

    def log_metrics(self):
        """Log metrics (to be implemented)."""
        # TODO: Implement metric logging

    def finish(self):
        """Finish pipeline execution (e.g., close logger)."""
        # TODO: Finalize logger, possibly wandb
