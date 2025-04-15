from __future__ import annotations  # noqa: D100
from abc import abstractmethod
from typing import Any, Dict, Optional

from framework3.base.base_clases import BaseFilter, XYData
from framework3.base.base_optimizer import BaseOptimizer

__all__ = ["BaseSplitter"]


class BaseSplitter(BaseFilter):
    @abstractmethod
    def start(
        self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
    ) -> Optional[XYData]:
        """
        Start the pipeline processing.

        Parameters:
        -----------
        x : XYData
            The input data to be processed.
        y : Optional[XYData]
            Optional target data.
        X_ : Optional[XYData]
            Optional additional input data.

        Returns:
        --------
        Optional[XYData]
            The processed data, if any.
        """
        ...

    @abstractmethod
    def log_metrics(self) -> None:
        """
        Log the metrics of the pipeline.

        This method should be implemented to record and possibly display
        the performance metrics of the pipeline.
        """
        ...

    @abstractmethod
    def split(self, pipeline: BaseFilter) -> None:
        """Escribe el docstring para este método.
        Optimize the pipeline based on the provided data.
        Parameters:
        -----------
        pipeline : BasePipeline
        The pipeline to be optimized.
        Returns:
        -------
        None
        """
        ...

    def verbose(self, value: bool) -> None:
        self._verbose = value
        self.pipeline.verbose(value)

    @abstractmethod
    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the metric based on the provided data.

        This method should be implemented by subclasses to calculate the specific metric.

        Parameters:
        -----------
        x_data : XYData
            The input data used for the prediction.
        y_true : XYData
            The ground truth or actual values.
        y_pred : XYData
            The predicted values.

        Returns:
        --------
        Float | np.ndarray
            The calculated metric value. This can be a single float or a numpy array,
            depending on the specific metric implementation.
        """
        ...

    def _pre_fit_wrapp(self, x: XYData, y: Optional[XYData]) -> float | None:
        return self._original_fit(x, y)

    def _pre_predict_wrapp(self, x: XYData) -> XYData:
        return self._original_predict(x)

    def optimizer(self, optimizer: BaseOptimizer) -> BaseOptimizer:
        optimizer.optimize(self)
        return optimizer

    def unwrap(self) -> BaseFilter:
        return self.pipeline
