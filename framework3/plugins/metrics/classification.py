from sklearn.metrics import f1_score, precision_score, recall_score
from framework3.base.base_types import Float
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseMetric
from framework3.container.container import Container

import numpy as np

__all__ = ["F1", "Precission", "Recall"]


@Container.bind()
class F1(BaseMetric):
    """
    F1 score metric for classification tasks.

    This class calculates the F1 score, which is the harmonic mean of precision and recall.
    It's particularly useful when you need a balance between precision and recall.

    Attributes:
        average (str): The type of averaging performed on the data. Default is 'weighted'.

    Example:
        ```python
        >>> from framework3.plugins.metrics.classification import F1
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> y_true = XYData(value=np.array([0, 1, 2, 0, 1, 2]))
        >>> y_pred = XYData(value=np.array([0, 2, 1, 0, 0, 1]))
        >>> x_data = XYData(value=np.array([1, 2, 3, 4, 5, 6]))
        >>>
        >>> f1_metric = F1(average='macro')
        >>> score = f1_metric.evaluate(x_data, y_true, y_pred)
        >>> print(f"F1 Score: {score}")
        ```
    """

    def __init__(self, average: str = "weighted"):
        """
        Initialize the F1 metric.

        Args:
            average (str): The type of averaging performed on the data. Default is 'weighted'.
        """
        super().__init__(average=average)
        self.average = average

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the F1 score.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (XYData): The ground truth (correct) target values.
            y_pred (XYData): The estimated targets as returned by a classifier.
            **kwargs (Any): Additional keyword arguments passed to sklearn's f1_score.

        Returns:
            (Float|np.ndarray): The F1 score or array of F1 scores if average is None.
        """
        if y_true is None:
            raise ValueError("Ground truth (y_true) must be provided.")
        return f1_score(
            y_true.value,
            y_pred.value,
            zero_division=0,
            average=self.average,  # type: ignore
        )  # type: ignore


@Container.bind()
class Precission(BaseMetric):
    """
    Precision metric for classification tasks.

    This class calculates the precision score, which is the ratio tp / (tp + fp) where tp is the number of true positives
    and fp the number of false positives.

    Attributes:
        average (str): The type of averaging performed on the data. Default is 'weighted'.

    Example:
        >>> from framework3.plugins.metrics.classification import Precission
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> y_true = XYData(value=np.array([0, 1, 2, 0, 1, 2]))
        >>> y_pred = XYData(value=np.array([0, 2, 1, 0, 0, 1]))
        >>> x_data = XYData(value=np.array([1, 2, 3, 4, 5, 6]))
        >>>
        >>> precision_metric = Precission(average='macro')
        >>> score = precision_metric.evaluate(x_data, y_true, y_pred)
        >>> print(f"Precision Score: {score}")
    """

    def __init__(self, average: str = "weighted"):
        """
        Initialize the Precision metric.

        Args:
            average (str): The type of averaging performed on the data. Default is 'weighted'.
        """
        super().__init__(average=average)
        self.average = average

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the precision score.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (XYData): The ground truth (correct) target values.
            y_pred (XYData): The estimated targets as returned by a classifier.
            **kwargs (Any): Additional keyword arguments passed to sklearn's precision_score.

        Returns:
            (Float|np.ndarray): The precision score or array of precision scores if average is None.
        """
        if y_true is None:
            raise ValueError("Ground truth (y_true) must be provided.")
        return precision_score(
            y_true.value,
            y_pred.value,
            zero_division=0,
            average=self.average,
            **kwargs,  # type: ignore
        )  # type: ignore


@Container.bind()
class Recall(BaseMetric):
    """
    Recall metric for classification tasks.

    This class calculates the recall score, which is the ratio tp / (tp + fn) where tp is the number of true positives
    and fn the number of false negatives.

    Attributes:
        average (str): The type of averaging performed on the data. Default is 'weighted'.

    Example:
    ```python
        >>> from framework3.plugins.metrics.classification import Recall
        >>> from framework3.base.base_types import XYData
        >>> import numpy as np
        >>>
        >>> y_true = XYData(value=np.array([0, 1, 2, 0, 1, 2]))
        >>> y_pred = XYData(value=np.array([0, 2, 1, 0, 0, 1]))
        >>> x_data = XYData(value=np.array([1, 2, 3, 4, 5, 6]))
        >>>
        >>> recall_metric = Recall(average='macro')
        >>> score = recall_metric.evaluate(x_data, y_true, y_pred)
        >>> print(f"Recall Score: {score}")
    ```
    """

    def __init__(self, average: str = "weighted"):
        """
        Initialize the Recall metric.

        Args:
            average (str): The type of averaging performed on the data. Default is 'weighted'.
        """
        super().__init__(average=average)
        self.average = average

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the recall score.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (XYData): The ground truth (correct) target values.
            y_pred (XYData): The estimated targets as returned by a classifier.
            **kwargs (Any): Additional keyword arguments passed to sklearn's recall_score.

        Returns:
            (Float|np.ndarray): The recall score or array of recall scores if average is None.
        """
        if y_true is None:
            raise ValueError("Ground truth (y_true) must be provided.")
        return recall_score(
            y_true.value,
            y_pred.value,
            zero_division=0,
            average=self.average,
            **kwargs,  # type: ignore
        )  # type: ignore
