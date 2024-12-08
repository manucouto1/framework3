from copy import deepcopy
from typing import Any, Dict, Sequence
from framework3.base import XYData, BaseFilter
from framework3.base import ParallelPipeline
from framework3.base.exceptions import NotTrainableFilterError
from framework3.container import Container
from framework3.utils.pyspark import PySparkMapReduce
from framework3.base.base_types import VData

import numpy as np

__all__ = ["HPCPipeline"]


@Container.bind()
class HPCPipeline(ParallelPipeline):
    """
    A pipeline that uses MapReduce to extract features in parallel using multiple filters.

    This pipeline applies a sequence of filters to the input data using a MapReduce approach,
    which allows for parallel processing and potentially improved performance on large datasets.

    Attributes:
        filters (Sequence[BaseFilter]): A sequence of filters to be applied to the input data.
        numSlices (int): The number of partitions to use in the MapReduce process.
        _map_reduce (PySparkMapReduce): The MapReduce implementation used for parallel processing.
    """

    def __init__(
        self,
        filters: Sequence[BaseFilter],
        app_name: str,
        master: str = "local",
        numSlices: int = 4,
    ):
        """
        Initialize the MapReduceFeatureExtractorPipeline.

        Args:
            filters (Sequence[BaseFilter]): A sequence of filters to be applied to the input data.
            app_name (str): The name of the Spark application.
            master (str, optional): The Spark master URL. Defaults to "local".
            numSlices (int, optional): The number of partitions to use in the MapReduce process. Defaults to 4.
        """
        super().__init__(filters=filters)
        self.filters = filters
        self.numSlices = numSlices
        self.app_name = app_name
        self.master = master

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
        """
        Start the pipeline by fitting the model and making predictions.

        Args:
            x (XYData): The input data.
            y (XYData | None): The target data, if available.
            X_ (XYData | None): Additional input data, if available.

        Returns:
            XYData | None: The predictions made by the pipeline.

        Raises:
            Exception: If an error occurs during the process.
        """
        try:
            self.fit(x, y)
            return self.predict(x)
        except Exception as e:
            # Handle the exception appropriately
            raise e

    def fit(self, x: XYData, y: XYData | None = None):
        """
        Fit the filters in the pipeline to the input data.

        This method applies the fit operation to all filters in parallel using MapReduce.

        Args:
            x (XYData): The input data to fit the filters on.
            y (XYData | None, optional): The target data, if available. Defaults to None.
        """

        def fit_function(filter):
            try:
                filter.fit(deepcopy(x), y)
            except NotTrainableFilterError:
                filter.init()
            return filter

        spark = PySparkMapReduce(self.app_name, self.master)
        # Apply fit in parallel to the filters
        rdd = spark.map(self.filters, fit_function, numSlices=self.numSlices)
        # Update the filters with the trained versions
        self.filters = rdd.collect()
        spark.stop()

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted filters.

        This method applies the predict operation to all filters in parallel using MapReduce,
        then combines the results.

        Args:
            x (XYData): The input data to make predictions on.

        Returns:
            XYData: The combined predictions from all filters.
        """

        def predict_function(filter: BaseFilter) -> VData:
            result: XYData = filter.predict(x)
            m_hash, _ = filter._get_model_key(x._hash)
            return XYData.ensure_dim(result.value)

        # Apply predict in parallel to the filters
        spark = PySparkMapReduce(self.app_name, self.master)
        spark.map(self.filters, predict_function, numSlices=self.numSlices)
        aux = spark.reduce(lambda x, y: np.hstack([x, y]))
        spark.stop()
        # Reduce the results
        return XYData.mock(aux)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline's performance.

        Args:
            x_data (XYData): The input data.
            y_true (XYData | None): The true target values, if available.
            y_pred (XYData): The predicted values.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        # Implement evaluation if necessary
        return {}

    def log_metrics(self):
        """
        Log metrics for the pipeline.

        This method can be implemented to log any relevant metrics during the pipeline's execution.
        """
        # Implement metric logging if necessary
        pass

    def finish(self):
        """
        Finish the pipeline's execution.

        This method is called to perform any necessary cleanup or finalization steps.
        """
