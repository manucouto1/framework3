import optuna

from typing import Any, Callable, Dict, Sequence, Union, cast
from framework3 import Container
from framework3.base import BasePlugin, XYData

from rich import print

from framework3.base.base_clases import BaseFilter
from framework3.base.base_optimizer import BaseOptimizer


@Container.bind()
class OptunaOptimizer(BaseOptimizer):
    def __init__(
        self,
        direction: str,
        n_trials: int = 2,
        load_if_exists: bool = False,
        reset_study: bool = False,
        pipeline: BaseFilter | None = None,
        study_name: str | None = None,
        storage: str | None = None,
    ):
        super().__init__(direction=direction, study_name=study_name, storage=storage)
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        self.pipeline = pipeline
        self.n_trials = n_trials
        self.load_if_exists = load_if_exists
        self.reset_study = reset_study

    def optimize(self, pipeline: BaseFilter):
        self.pipeline = pipeline

        if (
            self.reset_study
            and self.study_name is not None
            and self.storage is not None
        ):
            optuna.delete_study(study_name=self.study_name, storage=self.storage)

        self._study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
        )

    def exec(self, config: Dict[str, Any], x: XYData, y: XYData | None = None): ...

    def get_grid(self, aux: Dict[str, Any], f: Callable, init: bool = False):
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    self.get_grid(filter_config, f)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                self.get_grid(pipeline, f)
            case p_params:
                if "_grid" in aux:
                    for param, value in aux["_grid"].items():
                        value = f(param, value)
                        print(f"categorical param: {param}: {value}")
                        p_params.update({param: value})
                    # if init:
                    #     self._grid[aux["clazz"]] = aux["_grid"]

    def build_pipeline(
        self, dumped_pipeline: Dict[str, Any], f: Callable
    ) -> BaseFilter:
        self.get_grid(dumped_pipeline, f)
        print(dumped_pipeline)

        pipeline: BaseFilter = cast(
            BaseFilter, BasePlugin.build_from_dump(dumped_pipeline, Container.pif)
        )
        return pipeline

    def fit(self, x: XYData, y: XYData | None = None):
        if self.pipeline is not None:
            dumped_pipeline = self.pipeline.item_dump(include=["_grid"])

            def objective(trial) -> Union[float, Sequence[float]]:
                print(f"Trial: {trial.number}")

                def matcher(k, v):
                    match v:
                        case list():
                            return trial.suggest_categorical(k, v)
                        case dict():
                            if type(v["low"]) is int and type(v["high"]) is int:
                                return trial.suggest_int(k, v["low"], v["high"])
                            elif type(v["low"]) is float and type(v["high"]) is float:
                                return trial.suggest_float(k, v["low"], v["high"])
                            else:
                                raise ValueError(
                                    f"Inconsistent types in tuple: {k}: {v}"
                                )
                        case (min_v, max_v):
                            if type(min_v) is int and type(max_v) is int:
                                return trial.suggest_int(k, min_v, max_v)
                            elif type(min_v) is float and type(max_v) is float:
                                return trial.suggest_float(k, min_v, max_v)
                            else:
                                raise ValueError(
                                    f"Inconsistent types in tuple: {k}: {v}"
                                )
                        case _:
                            raise ValueError(f"Unsupported type in grid: {k}: {v}")

                pipeline: BaseFilter = self.build_pipeline(dumped_pipeline, matcher)

                match pipeline.fit(x, y):
                    case None:
                        return float(
                            next(
                                iter(
                                    pipeline.evaluate(
                                        x, y, pipeline.predict(x)
                                    ).values()
                                )
                            )
                        )
                    case float() as loss:
                        return loss
                    case _:
                        raise ValueError("Unsupported type in pipeline.fit")

            self._study.optimize(
                objective, n_trials=self.n_trials, show_progress_bar=True
            )

            best_params = self._study.best_params
            if best_params:
                print(f"Best params: {best_params}")
                pipeline = self.build_pipeline(
                    dumped_pipeline, lambda k, _: best_params[k]
                ).unwrap()
                pipeline.fit(x, y)
                self.pipeline = pipeline
            else:
                self.pipeline.unwrap().fit(x, y)
        else:
            raise ValueError("Pipeline must be defined before fitting")

    def predict(self, x: XYData) -> XYData:
        if self.pipeline is not None:
            return self.pipeline.predict(x)
        else:
            raise ValueError("Pipeline must be fitted before predicting")

    def log_metrics(self) -> None: ...
    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        if self.pipeline is not None:
            return self.pipeline.evaluate(x_data, y_true, y_pred)
        else:
            raise ValueError("Pipeline must be fitted before evaluating")

    def start(
        self, x: XYData, y: XYData | None, X_: XYData | None
    ) -> XYData | None: ...

    def finish(self) -> None: ...
