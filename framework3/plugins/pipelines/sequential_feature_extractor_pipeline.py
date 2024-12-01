from typing import List, Dict, Any, Sequence
from framework3.base import BasePipeline, XYData
import numpy as np

from framework3.base import BaseFilter
from framework3.container.container import Container

@Container.bind()
class SequentialFeatureExtractorPipeline(BasePipeline):
    """
    A module that combines multiple pipelines in parallel and constructs new features from their outputs.

    This class allows you to run multiple pipelines simultaneously on the same input data,
    and then combine their outputs to create new features.

    Attributes:
        pipelines (List[BasePipeline]): List of pipelines to be run in parallel.
        combiner_function (callable): A function that combines the outputs of the pipelines.
    """

    def __init__(self, filters: Sequence[BaseFilter]):
        """
        Initialize the CombinerPipeline.

        Args:
            pipelines (List[BasePipeline]): List of pipelines to be run in parallel.
            combiner_function (callable): A function that takes a list of XYData objects
                                          (outputs from pipelines) and returns a single XYData object.
        """
        super().__init__(filters=filters)
        self.filters = filters

    def init(self):
        """Initialize the pipeline (e.g., set up logging)."""
        # TODO: Initialize logger, possibly wandb

    def start(self, x: XYData, y: XYData|None, X_: XYData|None) -> XYData|None:
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
            print(f'Error during pipeline execution: {e}')
            raise e

    def fit(self, x: XYData, y: XYData|None = None):
        """
        Fit all pipelines in parallel.

        Args:
            x (XYData): Input data.
            y (XYData, optional): Target data. Defaults to None.
        """
        for filter in self.filters:
            filter.fit(x, y)

    def predict(self, x: XYData) -> XYData:
        """
        Run predictions on all pipelines in parallel and combine their outputs.

        Args:
            x (XYData): Input data.

        Returns:
            XYData: Combined output from all pipelines.
        """
        outputs: List[XYData] = [filter.predict(x) for filter in self.filters]
        return self.combine_features(outputs)
    
    def evaluate(self, x_data: XYData, y_true: XYData|None, y_pred: XYData) -> Dict[str, Any]:
        """
        Evaluate the pipeline using the provided metrics.

        Args:
            x (XYData): Input data.
            y (XYData): True target data.
            y_pred (XYData, optional): Predicted target data. If None, predictions will be made.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results for each metric.
        """
        
        results = {}
        for metric in self.metrics:
            results[metric.__class__.__name__] = metric.evaluate(x_data, y_true, y_pred)
        
        return results

    def log_metrics(self):
        """Log metrics (to be implemented)."""
        # TODO: Implement metric logging

    def finish(self):
        """Finish pipeline execution (e.g., close logger)."""
        # TODO: Finalize logger, possibly wandb


    @staticmethod
    def combine_features(pipeline_outputs: list[XYData]) -> XYData:
        """
        Default combiner function that concatenates features from all pipeline outputs.

        Args:
            pipeline_outputs (List[XYData]): List of outputs from each pipeline.

        Returns:
            XYData: Combined output with concatenated features.
        """
        return XYData.concat([output.value for output in pipeline_outputs], axis=-1)