from sklearn import datasets
from framework3 import (
    F1,
    Cached,
    F3Pipeline,
    KFoldSplitter,
    KnnFilter,
    Precission,
    Recall,
    StandardScalerPlugin,
)

from rich import print

from framework3.base.base_clases import XYData
from framework3.plugins.optimizer.grid_optimizer import GridOptimizer


def test_cached_with_grid_search():
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
                KnnFilter().grid({"n_neighbors": (2, 6)}),
            ],
            metrics=[F1(), Precission(), Recall()],
        )
        .splitter(
            KFoldSplitter(
                n_splits=2,
                shuffle=True,
                random_state=42,
            )
        )
        .optimizer(GridOptimizer(scorer=F1()))
    )

    wandb_pipeline.fit(X, y)

    assert len(list(wandb_pipeline._results.items())) == 2

    prediction = wandb_pipeline.predict(x=X)

    y_pred = XYData.mock(prediction.value)

    evaluate = wandb_pipeline.pipeline.evaluate(X, y, y_pred=y_pred)

    print(wandb_pipeline)

    print(evaluate)
