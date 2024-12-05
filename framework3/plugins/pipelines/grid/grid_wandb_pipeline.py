from typing import Any, Dict, Iterable, Literal
from framework3 import Container
from framework3.base import BaseMetric, GridPipeline
from framework3.base.base_types import XYData
from framework3.plugins.pipelines.sequential import F3Pipeline
from framework3.utils.wandb import WandbAgent, WandbSweepManager

import numpy as np
from rich import print
from tqdm import tqdm


@Container.bind()
class WandbPipeline(GridPipeline):
    def __init__(
        self,
        project: str,
        pipeline: F3Pipeline | None,
        scorer: BaseMetric,
        metrics: Iterable[BaseMetric],
        sweep_id: str | None = None,
        method: Literal["grid", "random", "bayes"] = "grid",
        splitter=None,
    ):
        super().__init__()
        if pipeline is None and sweep_id is None or project == "":
            raise ValueError("Either pipeline or sweep_id must be provided")
        self.project = project
        self.pipeline = pipeline
        self.metrics = metrics
        self.scorer = scorer
        self.sweep_id = sweep_id
        self.method = method
        self.splitter = splitter

    def exec(self, config: Dict[str, Any], x: XYData, y: XYData | None = None):
        print("-" * 200)
        print(f"*Config > {config}")
        print("-" * 200)

        def eval_config(x_train, y_train, x_test, y_test):
            pipeline = F3Pipeline(
                filters=list(
                    map(
                        lambda f: Container.ff[f](**config["filters"][f]),
                        config["order"],
                    )
                ),
                metrics=[self.scorer],
            )
            pipeline.fit(x_train, y_train)
            return pipeline.evaluate(x_test, y_test, pipeline.predict(x_test))

        if self.splitter is None:
            print("* No splitter provided, using default dataset")
            return eval_config(x, y, x, y)
        else:
            evals = []
            for train_indices, test_inidices in tqdm(
                self.splitter.split(x.value, y.value if y is not None else None),
                desc="Evaluating configurations..",
                leave=False,
            ):
                x_train, y_train = (
                    x.split(train_indices),
                    y.split(train_indices) if y is not None else None,
                )
                x_test, y_test = (
                    x.split(test_inidices),
                    y.split(test_inidices) if y is not None else None,
                )

                print("_" * 100)
                print(f"\n> Data split train {x_train} - {y_train}")
                print(f"> Data split test {x_test} - {y_test}\n")
                print("_" * 100)
                evals.append(
                    eval_config(x_train, y_train, x_test, y_test)[
                        self.scorer.__class__.__name__
                    ]
                )

            return {self.scorer.__class__.__name__: np.mean(evals)}

    def fit(self, x: XYData, y: XYData | None = None):
        if self.sweep_id is None and self.pipeline is not None:
            self.sweep_id = WandbSweepManager().create_sweep(
                self.pipeline, self.project, scorer=self.scorer, x=x, y=y
            )

        if self.sweep_id is not None:
            WandbAgent()(
                self.sweep_id, self.project, lambda config: self.exec(config, x, y)
            )
        else:
            raise ValueError("Either pipeline or sweep_id must be provided")

        winner = WandbSweepManager().get_best_config(
            self.project, self.sweep_id, self.scorer.__class__.__name__
        )

        self.pipeline = F3Pipeline(
            filters=list(
                map(lambda f: Container.ff[f](**winner["filters"][f]), winner["order"])
            ),
            metrics=[self.scorer],
        )
        self.pipeline.fit(x, y)

    def predict(self, x: XYData) -> XYData:
        if self.pipeline is not None:
            return self.pipeline.predict(x)
        else:
            raise ValueError("Pipeline must be fitted before predicting")

    def init(self) -> None: ...

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
        if self.pipeline is not None:
            return self.pipeline.start(x, y, X_)
        else:
            raise ValueError("Pipeline must be fitted before starting")

    def log_metrics(self) -> None:
        if self.pipeline is not None:
            self.pipeline.log_metrics()
        else:
            raise ValueError("Pipeline must be fitted before logging metrics")

    def evaluate(self, x: XYData, y: XYData | None, X_: XYData) -> Dict[str, Any]:
        return self.pipeline.evaluate(x, y, X_) if self.pipeline is not None else {}

    def finish(self) -> None: ...
