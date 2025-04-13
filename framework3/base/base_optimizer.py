from __future__ import annotations  # noqa: D100
from abc import abstractmethod
from typing import Optional

from framework3.base.base_clases import BaseFilter, XYData

__all__ = ["BaseOptimizer"]


class BaseOptimizer(BaseFilter):
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
    def optimize(self, pipeline: BaseFilter) -> None:
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
