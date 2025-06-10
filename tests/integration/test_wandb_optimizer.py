from unittest.mock import MagicMock
from rich import print
from sklearn import datasets
import pytest
import typeguard

from framework3 import (
    F1,
    Cached,
    F3Pipeline,
    KnnFilter,
    Precission,
    Recall,
    StandardScalerPlugin,
    WandbOptimizer,
)
from framework3.plugins.metrics.classification import XYData
from framework3.plugins.splitter.cross_validation_splitter import KFoldSplitter


def test_wandb_pipeline_init_raises_value_error():
    from framework3.base import BaseMetric

    with pytest.raises(
        ValueError, match="Either pipeline or sweep_id must be provided"
    ):
        WandbOptimizer(
            project="test_project",
            pipeline=None,
            sweep_id=None,
            scorer=MagicMock(spec=BaseMetric),
        ).fit(XYData.mock([]), XYData.mock([]))


def test_wandb_pipeline_init_raises_value_error_for_invalid_project():
    from framework3.base import BaseMetric

    with pytest.raises(
        ValueError, match="Either pipeline or sweep_id must be provided"
    ):
        WandbOptimizer(
            project="",
            pipeline=None,
            sweep_id=None,
            scorer=MagicMock(spec=BaseMetric),
        ).fit(XYData.mock([]), XYData.mock([]))

    with pytest.raises(typeguard.TypeCheckError):
        WandbOptimizer(
            project=None,  # type: ignore
            pipeline=MagicMock(),
            sweep_id=None,
            scorer=MagicMock(spec=BaseMetric),
        )


def test_wandb_pipeline_init_with_valid_parameters():
    from framework3.base import BaseMetric

    mock_pipeline = MagicMock()
    mock_scorer = MagicMock(spec=BaseMetric)

    wandb_pipeline = WandbOptimizer(
        project="test_project",
        pipeline=mock_pipeline,
        sweep_id=None,
        scorer=mock_scorer,
    )

    assert wandb_pipeline.project == "test_project"
    assert wandb_pipeline.pipeline == mock_pipeline
    assert wandb_pipeline.sweep_id is None
    assert wandb_pipeline.scorer == mock_scorer


def test_wandb_pipeline_init_and_fit():
    iris = datasets.load_iris()

    X = XYData(
        _hash="Iris X data",
        _path="/datasets",
        _value=iris.data,  # type: ignore
    )
    y = XYData(
        _hash="Iris y data",
        _path="/datasets",
        _value=iris.target,  # type: ignore
    )

    wandb_pipeline = (
        F3Pipeline(
            filters=[
                Cached(StandardScalerPlugin()),
                KnnFilter().grid({"n_neighbors": [3, 5]}),
            ],
            metrics=[
                F1(average="weighted"),
                Precission(average="weighted"),
                Recall(average="weighted"),
            ],
        )
        .splitter(
            KFoldSplitter(
                n_splits=2,
                shuffle=True,
                random_state=42,
            )
        )
        .optimizer(
            WandbOptimizer(
                project="test_project",
                sweep_id=None,
                scorer=F1(average="weighted"),
            )
        )
    )

    print("______________________PIPELINE_____________________")
    print(wandb_pipeline)
    print("_____________________________________________________")

    assert wandb_pipeline.sweep_id is None

    try:
        wandb_pipeline.fit(X, y)
        prediction = wandb_pipeline.predict(x=X)

        y_pred = XYData.mock(prediction.value)

        evaluate = wandb_pipeline.evaluate(X, y, y_pred)

        print(wandb_pipeline)

        print(evaluate)

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        assert False
