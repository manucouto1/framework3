from typing import Any, Callable, Dict, List, Literal
import wandb

from framework3.base import BaseMetric, XYData
from framework3.plugins.pipelines.sequential import F3Pipeline


class WandbSweepManager:
    @staticmethod
    def generate_config_for_pipeline(pipepline: F3Pipeline) -> Dict[str, Any]:
        """
        Generate a Weights & Biases sweep configuration from a dumped pipeline.

        Args:
            dumped_pipeline (Dict[str, Any]): The result of pipeline.item_dump(include=["_grid"])

        Returns:
            Dict[str, Any]: A wandb sweep configuration
        """
        sweep_config: Dict[str, Dict[str, Dict[str, Any]]] = {
            "parameters": {"filters": {"parameters": {}}, "order": {"value": []}}
        }

        dumped_pipeline = pipepline.item_dump(include=["_grid"])
        for filter_config in dumped_pipeline["params"]["filters"]:
            if "_grid" in filter_config:
                filter_config["params"].update(**filter_config["_grid"])

            f_config = {}
            for k, v in filter_config["params"].items():
                if type(v) is list:
                    f_config[k] = {"values": v}
                elif type(v) is dict:
                    f_config[k] = v
                else:
                    f_config[k] = {"value": v}

            sweep_config["parameters"]["filters"]["parameters"][
                str(filter_config["clazz"])
            ] = {"parameters": f_config}
            sweep_config["parameters"]["order"]["value"].append(filter_config["clazz"])

        return sweep_config

    def create_sweep(
        self,
        pipeline: F3Pipeline,
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
