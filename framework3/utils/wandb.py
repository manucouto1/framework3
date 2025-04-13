from typing import Any, Callable, Dict, List, Literal
import wandb
from rich import print
from framework3.base import BaseMetric, XYData, BaseFilter


class WandbSweepManager:
    @staticmethod
    def get_grid(aux: Dict[str, Any], config: Dict[str, Any]):
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    WandbSweepManager.get_grid(filter_config, config)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                WandbSweepManager.get_grid(pipeline, config)
            case p_params:
                if "_grid" in aux:
                    f_config = {}
                    for param, value in aux["_grid"].items():
                        print(f"categorical param: {param}: {value}")
                        p_params.update({param: value})
                        match type(value):
                            case list():
                                f_config[param] = {"values": value}
                            case dict():
                                f_config[param] = value
                            case _:
                                f_config[param] = {"values": value}
                    if len(f_config) > 0:
                        config["parameters"]["filters"]["parameters"][
                            str(aux["clazz"])
                        ] = {"parameters": f_config}

    @staticmethod
    def generate_config_for_pipeline(pipepline: BaseFilter) -> Dict[str, Any]:
        """
        Generate a Weights & Biases sweep configuration from a dumped pipeline.

        Args:
            dumped_pipeline (Dict[str, Any]): The result of pipeline.item_dump(include=["_grid"])

        Returns:
            Dict[str, Any]: A wandb sweep configuration
        """
        sweep_config: Dict[str, Dict[str, Dict[str, Any]]] = {
            "parameters": {"filters": {"parameters": {}}, "pipeline": {"value": {}}}
        }

        dumped_pipeline = pipepline.item_dump(include=["_grid"])
        # for filter_config in dumped_pipeline["params"]["filters"]:
        #     if "_grid" in filter_config:
        #         filter_config["params"].update(**filter_config["_grid"])

        #         f_config = {}
        #         for k, v in filter_config["_grid"].items():
        #             if type(v) is list:
        #                 f_config[k] = {"values": v}
        #             elif type(v) is dict:
        #                 f_config[k] = v
        #             else:
        #                 f_config[k] = {"value": v}

        #         if len(f_config) > 0:
        #             sweep_config["parameters"]["filters"]["parameters"][
        #                 str(filter_config["clazz"])
        #             ] = {"parameters": f_config}

        WandbSweepManager.get_grid(dumped_pipeline, sweep_config)

        sweep_config["parameters"]["pipeline"]["value"] = dumped_pipeline

        return sweep_config

    def create_sweep(
        self,
        pipeline: BaseFilter,
        project_name: str,
        scorer: BaseMetric,
        x: XYData,
        y: XYData | None = None,
    ) -> str:
        sweep_config = WandbSweepManager.generate_config_for_pipeline(pipeline)
        sweep_config["method"] = "grid"
        sweep_config["parameters"]["x_dataset"] = {"value": x._hash}
        sweep_config["parameters"]["y_dataset"] = (
            {"value": y._hash} if y is not None else {"value": "None"}
        )
        sweep_config["metric"] = {
            "name": scorer.__class__.__name__,
            "goal": "maximize" if scorer.higher_better else "minimize",
        }
        print(sweep_config)
        return wandb.sweep(sweep_config, project=project_name)  # type: ignore

    def get_sweep(self, project_name, sweep_id):
        sweep = wandb.Api().sweep(f"citius-irlab/{project_name}/sweeps/{sweep_id}")  # type: ignore
        return sweep

    def get_best_config(self, project_name, sweep_id, order):
        sweep = self.get_sweep(project_name, sweep_id)
        winner_run = sweep.best_run(order=order)
        return dict(winner_run.config)

    def restart_sweep(self, sweep, states: List[str] | Literal["all"] = "all"):
        # Eliminar todas las ejecuciones fallidas
        for run in sweep.runs:
            if run.state in states or states == "all":
                run.delete()
                print("Deleting run:", run.id)

    def init(self, group: str, name: str, reinit=True):
        run = wandb.init(group=group, name=name, reinit=reinit)  # type: ignore
        return run


class WandbAgent:
    @staticmethod
    def __call__(sweep_id: str, project: str, function: Callable):
        wandb.agent(  # type: ignore
            sweep_id,
            function=lambda: {
                wandb.init(reinit=True),  # type: ignore
                wandb.log(function(dict(wandb.config))),  # type: ignore
            },
            project=project,
        )  # type: ignore
        wandb.teardown()  # type: ignore


class WandbRunLogger: ...
