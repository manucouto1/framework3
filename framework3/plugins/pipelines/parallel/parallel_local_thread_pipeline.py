from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Dict, Sequence
from tqdm import tqdm

from framework3.base import XYData, BaseFilter, ParallelPipeline
from framework3.base.exceptions import NotTrainableFilterError
from framework3.container import Container
from framework3.base.base_types import VData

__all__ = ["LocalThreadPipeline"]


# @contextlib.contextmanager
# def suppress_stdout():
#     with open(os.devnull, "w") as devnull:
#         with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
#             yield


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

    def fit(self, x: XYData, y: XYData | None = None):
        def fit_function(filt: BaseFilter) -> BaseFilter:
            try:
                filt.fit(deepcopy(x), y)
            except NotTrainableFilterError:
                filt.init()
            return filt

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            self.filters = list(
                tqdm(
                    executor.map(fit_function, self.filters),
                    total=len(self.filters),
                    desc="Fitting",
                )
            )

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
        return {}

    def log_metrics(self):
        pass

    def finish(self):
        pass
