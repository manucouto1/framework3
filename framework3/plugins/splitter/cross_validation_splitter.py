from typing import Any, Dict, Optional
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from framework3 import Container
from framework3.base.base_clases import BaseFilter
from framework3.base.base_splitter import BaseSplitter
from framework3.base.base_types import XYData


@Container.bind()
class KFoldSplitter(BaseSplitter):
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        pipeline: BaseFilter | None = None,
        # evaluator: BaseMetric | None = None
    ):
        super().__init__()
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._kfold = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        self.pipeline = pipeline
        # self.evaluator = evaluator

    def split(self, pipeline: BaseFilter):
        self.pipeline = pipeline

    def fit(self, x: XYData, y: XYData | None) -> Optional[float]:
        X = x.value
        if y is None:  # type: ignore
            raise ValueError("y must be provided for KFold split")

        Y = y.value

        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before splitting")

        losses = []
        splits = self._kfold.split(X)
        for train_idx, val_idx in tqdm(splits, total=self._kfold.get_n_splits(X)):
            X_train = XYData(
                _hash=f"{x._hash}_{train_idx}",
                _path=f"{x._path}_{train_idx}",
                _value=X[train_idx],
            )
            X_val = XYData(
                _hash=f"{x._hash}_{val_idx}",
                _path=f"{x._path}_{val_idx}",
                _value=X[val_idx],
            )
            y_train = XYData(
                _hash=f"{y._hash}_{train_idx}",
                _path=f"{y._path}_{train_idx}",
                _value=Y[train_idx],
            )
            y_val = XYData(
                _hash=f"{y._hash}_{val_idx}",
                _path=f"{y._path}_{val_idx}",
                _value=Y[val_idx],
            )
            self.pipeline.fit(X_train, y_train)

            _y = self.pipeline.predict(X_val)

            loss = self.pipeline.evaluate(X_val, y_val, _y)
            losses.append(float(next(iter(loss.values()))))

        return float(np.mean(losses) if losses else 0.0)

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

    def predict(self, x: XYData) -> XYData:
        # X = x.value
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before prediction")

        return self.pipeline.predict(x)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before evaluation")
        return self.pipeline.evaluate(x_data, y_true, y_pred)

    def log_metrics(self) -> None: ...
