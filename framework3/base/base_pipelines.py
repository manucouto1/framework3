from abc import abstractmethod
from typing import Optional
from framework3.base.base_clases import BaseFilter
from framework3.base.base_types import XYData


__all__ = ["BasePipeline", "SequentialPipeline", "ParallelPipeline", "GridPipeline"]


class BasePipeline(BaseFilter):
    """
    Base class for implementing pipeline structures in the framework.

    This abstract class extends BaseFilter and defines the interface for pipeline operations.
    Subclasses should implement the abstract methods to provide specific pipeline functionality.

    Example:
        ```python
        from framework3.base.base_clases import BasePipeline
        from framework3.base.base_types import XYData

        class MyCustomPipeline(BasePipeline):
            def fit(self, x: XYData, y: Optional[XYData]) -> None:
                # Implement fitting logic here
                pass

            def predict(self, x: XYData) -> XYData:
                # Implement prediction logic here
                pass

            # Implement other required methods...

        ```
    """

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
    def finish(self) -> None:
        """
        Finish the pipeline processing.

        This method should be implemented to perform any necessary cleanup
        or finalization steps after the pipeline has completed its processing.
        """
        ...

    def init(self):
        super().init()
        for filter in self.filters:
            filter.init()


class SequentialPipeline(BasePipeline):
    """
    Base class for orchestrators that manage complex data flows and combinations of filters/pipelines.
    """

    def _pre_fit(self, x: XYData, y: Optional[XYData]):
        m_hash, m_str = self._get_model_key(
            data_hash=f'{x._hash}, {y._hash if y is not None else ""}'
        )
        m_path = f"{self._get_model_name()}/{m_hash}"

        self._m_hash = m_hash
        self._m_path = m_path
        self._m_str = m_str

        new_x = x

        for filter in self.filters:
            filter._pre_fit(new_x, y)
            new_x = filter._pre_predict(new_x)

        return m_hash, m_path, m_str

    def _pre_predict(self, x: XYData):
        if not self._m_hash or not self._m_path or not self._m_str:
            raise ValueError("Cached filter model not trained or loaded")

        aux_x = x
        for filter in self.filters:
            aux_x = filter._pre_predict(aux_x)
        return aux_x


class ParallelPipeline(BasePipeline): ...


class GridPipeline(BasePipeline): ...
