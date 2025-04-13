from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np

from framework3.base import XYData, BaseFilter, ParallelPipeline
from framework3.base.exceptions import NotTrainableFilterError
from framework3.container import Container
from framework3.base.base_types import VData

__all__ = ["LocalThreadPipeline"]


@Container.bind()
class LocalThreadPipeline(ParallelPipeline):
    """
    A pipeline that runs filters in parallel using local multithreading.

    This is a lightweight version of HPCPipeline, using ThreadPoolExecutor
    instead of a distributed Spark cluster.
    """

    def __init__(self, filters: Sequence[BaseFilter], num_threads: int = 4):
        super().__init__(filters=filters)
        self.filters = filters
        self.num_threads = num_threads

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
        try:
            self.fit(x, y)
            return self.predict(x)
        except Exception as e:
            raise e

    def fit(self, x: XYData, y: Optional[XYData]) -> Optional[float]:
        def fit_function(filt: BaseFilter) -> Tuple[BaseFilter, Optional[float]]:
            loss = None
            try:
                loss = filt.fit(deepcopy(x), y)
            except NotTrainableFilterError:
                filt.init()
            return filt, loss

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            self.filters, losses = list(
                zip(
                    *tqdm(
                        executor.map(fit_function, self.filters),
                        total=len(self.filters),
                        desc="Fitting",
                    )
                )
            )
            return float(np.mean(losses)) if losses else None

    def predict(self, x: XYData) -> XYData:
        def predict_function(filt: BaseFilter) -> VData:
            result: XYData = filt.predict(x)
            return XYData.ensure_dim(result.value)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(
                tqdm(
                    executor.map(predict_function, self.filters),
                    total=len(self.filters),
                    desc="Predicting",
                )
            )

        combined = XYData.concat(results)
        return combined

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

    def log_metrics(self):
        pass

    def finish(self):
        pass
