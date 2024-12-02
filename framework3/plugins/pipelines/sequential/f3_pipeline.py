import functools
from typing import Dict, List, Optional, cast

from pydantic import ConfigDict
from framework3.base import XYData
from framework3.base import BaseFilter, BaseMetric
from framework3.base import SequentialPipeline
from framework3.container import Container

from rich import print as rprint

__all__ = ['F3Pipeline']

@Container.bind()
class F3Pipeline(SequentialPipeline):
    """
    F3Pipeline is a flexible pipeline implementation for machine learning workflows.

    This class allows you to chain multiple plugins (filters) together and apply metrics
    for evaluation. It supports fitting, predicting, and evaluating data through the pipeline.

    Attributes:
        plugins (List[BasePlugin]): List of plugins (filters) to be applied in the pipeline.
        metrics (List[BaseMetric]): List of metrics to evaluate the pipeline's performance.
        overwrite (bool): Whether to overwrite existing data in storage.
        store (bool): Whether to store intermediate results.
        log (bool): Whether to log pipeline operations.

    Example:
        ```python
        >>> from framework3.plugins.filters.transformation import PCAPlugin
        >>> from framework3.plugins.filters.clasification import ClassifierSVMPlugin
        >>> from framework3.plugins.metrics.classification import F1, Precision, Recall
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>> 
        >>> # Create a pipeline with PCA and SVM
        >>> pipeline = F3Pipeline(
        ...     plugins=[
        ...         PCAPlugin(n_components=2),
        ...         ClassifierSVMPlugin(kernel='rbf', C=1.0)
        ...     ],
        ...     metrics=[F1(), Precision(), Recall()]
        ... )
        >>> 
        >>> # Prepare some dummy data
        >>> X = XYData(value=np.random.rand(100, 10))
        >>> y = XYData(value=np.random.randint(0, 2, 100))
        >>> 
        >>> # Fit the pipeline
        >>> pipeline.fit(X, y)
        >>> 
        >>> # Make predictions
        >>> y_pred = pipeline.predict(X)
        >>> 
        >>> # Evaluate the pipeline
        >>> results = pipeline.evaluate(X, y, y_pred)
        >>> print(results)
        ```
    """

    model_config = ConfigDict(extra='allow')

    def __init__(self, filters: List[BaseFilter], metrics: List[BaseMetric]=[], overwrite: bool = False, store: bool = False, log: bool = False) -> None:
        """
        Initialize the F3Pipeline.

        Args:
            filters (List[BaseFilter]): List of plugins to be applied in the pipeline.
            metrics (List[BaseMetric]): List of metrics for evaluation.
            overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.
            store (bool, optional): Whether to store intermediate results. Defaults to False.
            log (bool, optional): Whether to log pipeline operations. Defaults to False.
        """
        super().__init__(filters=filters, metrics=metrics, overwrite=overwrite, store=store, log=log)
        self.filters: List[BaseFilter] = filters
        self.metrics: List[BaseMetric] = metrics
        self.overwrite = overwrite
        self.store = store
        self.log = log
        self._filters: List[BaseFilter] = []
    

    def init(self):
        """Initialize the pipeline (e.g., set up logging)."""
        # TODO: Initialize logger, possibly wandb

    def start(self, x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
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

    def log_metrics(self):
        """Log metrics (to be implemented)."""
        # TODO: Implement metric logging

    def finish(self):
        """Finish pipeline execution (e.g., close logger)."""
        # TODO: Finalize logger, possibly wandb

    def fit(self, x: XYData, y: Optional[XYData]):
        """
        Fit the pipeline to the input data.

        Args:
            x (XYData): Input data.
            y (Optional[XYData]): Target data.
        """
        rprint('_'*100)
        rprint('Fitting pipeline...')
        rprint('*'*100)
        if self._filters:
            self._filters = []

        for filters in self.filters:
            rprint(f'\n* {filters}:')
            filters.fit(x, y)
            x = filters.predict(x)
            self._filters.append(filters)
    
    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted pipeline.

        Args:
            x (XYData): Input data for prediction.

        Returns:
            XYData: Prediction results.

        Raises:
            ValueError: If no filters have been trained yet.
        """
        print("#### Quinto llama a los predict de los filtros desde el pipeline")
        if not self._filters:
            raise ValueError('No filters have been trained yet')
        rprint('_'*100)
        rprint('Predicting pipeline...')
        rprint('*'*100)

        for filter_ in self._filters:
            rprint(f'\n* {filter_}')
            x = filter_.predict(x)
            
        return x
    
    def evaluate(self, x_data: XYData, y_true: XYData|None, y_pred: XYData) -> Dict[str, float]:
        """
        Evaluate the pipeline using the specified metrics.

        Args:
            x_data (XYData): Input data.
            y_true (XYData): True target values.
            y_pred (XYData): Predicted values.

        Returns:
            Dict[str, float]: A dictionary of metric names and their corresponding values.
        """
        rprint('Evaluating pipeline...')
        evaluations = {}
        for metric in self.metrics:
            evaluations[metric.__class__.__name__] = metric.evaluate(x_data, y_true, y_pred)
        return evaluations
    