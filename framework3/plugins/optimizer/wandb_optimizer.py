from typing import Any, Dict, cast
from framework3 import Container
from framework3.base import BaseMetric
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.base.base_optimizer import BaseOptimizer
from framework3.base.base_types import XYData
from framework3.utils.wandb import WandbAgent, WandbSweepManager

from rich import print

__all__ = ["WandbOptimizer"]


@Container.bind()
class WandbOptimizer(BaseOptimizer):
    def __init__(
        self,
        project: str,
        scorer: BaseMetric,
        pipeline: BaseFilter | None = None,
        sweep_id: str | None = None,
    ):
        super().__init__()
        self.project = project
        self.scorer = scorer
        self.sweep_id = sweep_id
        self.pipeline = pipeline

    def optimize(self, pipeline: BaseFilter) -> None:
        self.pipeline = pipeline
        self.pipeline.verbose(False)

    def get_grid(self, aux: Dict[str, Any], config: Dict[str, Any]) -> None:
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    self.get_grid(filter_config, config)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                self.get_grid(pipeline, config)
            case p_params:
                if "_grid" in aux:
                    for param, value in aux["_grid"].items():
                        p_params.update({param: config[aux["clazz"]][param]})

    def exec(self, config: Dict[str, Any], x: XYData, y: XYData | None = None):
        if self.pipeline is None and self.sweep_id is None or self.project == "":
            raise ValueError("Either pipeline or sweep_id must be provided")

        self.get_grid(config["pipeline"], config["filters"])

        pipeline: BaseFilter = cast(
            BaseFilter, BasePlugin.build_from_dump(config["pipeline"], Container.pif)
        )

        pipeline.verbose(False)

        match pipeline.fit(x, y):
            case None:
                losses = pipeline.evaluate(x, y, pipeline.predict(x))

                loss = losses.get(self.scorer.__class__.__name__, 0.0)

                return {self.scorer.__class__.__name__: float(loss)}
            case float() as loss:
                return {self.scorer.__class__.__name__: loss}
            case _:
                raise ValueError("Unexpected return type from pipeline.fit()")

    def fit(self, x: XYData, y: XYData | None = None) -> None:
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

        print(winner)

        self.get_grid(winner["pipeline"], winner["filters"])
        self.pipeline = cast(
            BaseFilter, BasePlugin.build_from_dump(winner["pipeline"], Container.pif)
        )

        self.pipeline.unwrap().fit(x, y)

    def predict(self, x: XYData) -> XYData:
        if self.pipeline is not None:
            return self.pipeline.predict(x)
        else:
            raise ValueError("Pipeline must be fitted before predicting")

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
        if self.pipeline is not None:
            return self.pipeline.start(x, y, X_)
        else:
            raise ValueError("Pipeline must be fitted before starting")

    def log_metrics(self) -> None:
        if self.pipeline is not None:
            return self.pipeline.log_metrics()
        else:
            raise ValueError("Pipeline must be fitted before logging metrics")

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        return (
            self.pipeline.evaluate(x_data, y_true, y_pred)
            if self.pipeline is not None
            else {}
        )

    def finish(self) -> None: ...
