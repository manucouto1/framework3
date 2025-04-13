from unittest import mock
from unittest.mock import ANY, MagicMock, patch

import pytest
import typeguard
import numpy as np

from framework3 import WandbOptimizer


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
            metrics=[MagicMock(spec=BaseMetric)],
        )


def test_wandb_pipeline_init_raises_value_error_for_invalid_project():
    from framework3.base import BaseMetric

    with pytest.raises(
        ValueError, match="Either pipeline or sweep_id must be provided"
    ):
        WandbOptimizer(
            project="",
            pipeline=MagicMock(),
            sweep_id=None,
            scorer=MagicMock(spec=BaseMetric),
            metrics=[MagicMock(spec=BaseMetric)],
        )

    with pytest.raises(typeguard.TypeCheckError):
        WandbOptimizer(
            project=None,  # type: ignore
            pipeline=MagicMock(),
            sweep_id=None,
            scorer=MagicMock(spec=BaseMetric),
            metrics=[MagicMock(spec=BaseMetric)],
        )


def test_wandb_pipeline_init_with_valid_parameters():
    from framework3.base import BaseMetric

    mock_pipeline = MagicMock()
    mock_scorer = MagicMock(spec=BaseMetric)
    mock_metrics = [MagicMock(spec=BaseMetric)]

    wandb_pipeline = WandbOptimizer(
        project="test_project",
        pipeline=mock_pipeline,
        sweep_id=None,
        scorer=mock_scorer,
        metrics=mock_metrics,
    )

    assert wandb_pipeline.project == "test_project"
    assert wandb_pipeline.pipeline == mock_pipeline
    assert wandb_pipeline.sweep_id is None
    assert wandb_pipeline.scorer == mock_scorer
    assert wandb_pipeline.metrics == mock_metrics


def test_wandb_pipeline_init_and_fit():
    mock_wandb = MagicMock()
    mock_x = MagicMock()
    mock_y = MagicMock()
    with (
        patch(
            "framework3.plugins.pipelines.grid.grid_wandb_pipeline.WandbSweepManager"
        ) as mock_sweep_manager,
        patch(
            "framework3.plugins.pipelines.grid.grid_wandb_pipeline.WandbAgent"
        ) as mock_agent,
    ):
        from framework3 import (
            F1,
            F3Pipeline,
            StandardScalerPlugin,
            WandbOptimizer,
            XYData,
        )
        from framework3.base import BaseMetric
        from main import KnnFilter

        mock_scorer = MagicMock(spec=BaseMetric)
        mock_metrics = [MagicMock(spec=BaseMetric)]

        real_pipeline = F3Pipeline(
            filters=[StandardScalerPlugin(), KnnFilter(n_neighbors=3)], metrics=[F1()]
        )

        mock_scorer = F1()
        mock_metrics = [F1()]
        mock_wandb = MagicMock()

        # Create some dummy data
        mock_x = XYData.mock(np.random.rand(100, 5))
        mock_y = XYData.mock(np.random.randint(0, 2, 100))

        # Initialize WandbPipeline
        wandb_pipeline = WandbOptimizer(
            project="test_project",
            pipeline=real_pipeline,
            sweep_id=None,
            scorer=mock_scorer,
            metrics=mock_metrics,
        )

        assert wandb_pipeline.sweep_id is None
        mock_wandb.sweep.assert_not_called()

        # Mock the create_sweep method
        mock_sweep_manager.return_value.create_sweep.return_value = "new_sweep_id"
        mock_sweep_manager.return_value.get_best_config.return_value = {"order": []}
        mock_sweep_manager.return_value.get_best_config.side_effect = (
            lambda *args, **kwargs: {"order": []}
        )

        # Call fit method
        wandb_pipeline.fit(mock_x, mock_y)

        # Verify that create_sweep was called during fit
        mock_sweep_manager.return_value.create_sweep.assert_called_once_with(
            ANY, "test_project", scorer=mock_scorer, x=mock_x, y=mock_y
        )

        # Verify that the sweep_id was set after fit
        assert wandb_pipeline.sweep_id == "new_sweep_id"

        # Verify that WandbAgent was called
        mock_agent.assert_called_once()
        mock_agent.return_value.assert_called_once_with(
            "new_sweep_id", "test_project", mock.ANY
        )

        # Verify that get_best_config was called
        mock_sweep_manager.return_value.get_best_config.assert_called_once_with(
            "test_project", "new_sweep_id", mock_scorer.__class__.__name__
        )
