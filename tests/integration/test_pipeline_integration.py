import numpy as np
from sklearn import datasets
from framework3.base.base_types import XYData
from framework3.container import Container
from framework3.plugins.filters.classification.svm import ClassifierSVMPlugin
from framework3.plugins.filters.grid_search.cv_grid_search import GridSearchCVPlugin
from framework3.plugins.filters.transformation.pca import PCAPlugin
from framework3.plugins.metrics.classification import F1, Precission, Recall
from framework3.plugins.pipelines.sequential.f3_pipeline import F3Pipeline


def test_pipeline_iris_dataset():
    iris = datasets.load_iris()

    pipeline = F3Pipeline(
        filters=[
            PCAPlugin(n_components=1),
            GridSearchCVPlugin(
                ClassifierSVMPlugin,
                ClassifierSVMPlugin.item_grid(C=[1.0, 10], kernel=["rbf"]),
                scoring="f1_weighted",
                cv=2,
            ),
        ],
        metrics=[F1(), Precission(), Recall()],
    )

    X = XYData(
        _hash="Iris X data",
        _path=Container.storage.get_root_path(),
        _value=iris.data,  # type: ignore
    )
    y = XYData(
        _hash="Iris y data",
        _path=Container.storage.get_root_path(),
        _value=iris.target,  # type: ignore
    )

    pipeline.fit(X, y)
    prediction = pipeline.predict(x=X)
    print(prediction.value)
    evaluate = pipeline.evaluate(X, y, prediction)

    assert isinstance(prediction.value, np.ndarray)
    assert prediction.value.shape == (150,)
    assert isinstance(evaluate, dict)
    assert "F1" in evaluate
    assert "Precission" in evaluate
    assert "Recall" in evaluate
    assert all(0 <= score <= 1 for score in evaluate.values())


def test_pipeline_different_feature_counts():
    # Create datasets with different numbers of features
    iris = datasets.load_iris()

    X_full = XYData(
        _hash="Iris X data",
        _path=Container.storage.get_root_path(),
        _value=iris.data,  # type: ignore
    )
    X_reduced = XYData(
        _hash="Iris X reduced data",
        _path=Container.storage.get_root_path(),
        _value=iris.data[:, :3],  # type: ignore
    )

    y = XYData(
        _hash="Iris y data",
        _path=Container.storage.get_root_path(),
        _value=iris.target,  # type: ignore
    )

    pipeline = F3Pipeline(
        filters=[
            PCAPlugin(n_components=1),
            GridSearchCVPlugin(
                ClassifierSVMPlugin,
                ClassifierSVMPlugin.item_grid(C=[1.0, 10], kernel=["rbf"]),
                scoring="f1_weighted",
                cv=2,
            ),
        ],
        metrics=[F1(), Precission(), Recall()],
    )

    # Test with full dataset
    pipeline.fit(X_full, y=y)
    prediction_full = pipeline.predict(x=X_full)

    evaluate_full = pipeline.evaluate(X_full, y, y_pred=prediction_full)

    # Test with reduced dataset

    pipeline.fit(X_reduced, y)
    prediction_reduced = pipeline.predict(x=X_reduced)
    evaluate_reduced = pipeline.evaluate(X_reduced, y, y_pred=prediction_reduced)

    assert isinstance(prediction_full.value, np.ndarray)
    assert isinstance(prediction_reduced.value, np.ndarray)
    assert prediction_full.value.shape == prediction_reduced.value.shape == (150,)
    assert isinstance(evaluate_full, dict)
    assert isinstance(evaluate_reduced, dict)
    assert (
        set(evaluate_full.keys())
        == set(evaluate_reduced.keys())
        == {"F1", "Precission", "Recall"}
    )
    assert all(0 <= score <= 1 for score in evaluate_full.values())
    assert all(0 <= score <= 1 for score in evaluate_reduced.values())


def test_grid_search_with_specified_parameters():
    iris = datasets.load_iris()
    X = XYData(
        _hash="Iris X data",
        _path=Container.storage.get_root_path(),
        _value=iris.data,  # type: ignore
    )
    y = XYData(
        _hash="Iris y data",
        _path=Container.storage.get_root_path(),
        _value=iris.target,  # type: ignore
    )

    pipeline = F3Pipeline(
        filters=[
            PCAPlugin(n_components=1),
            GridSearchCVPlugin(
                filterx=ClassifierSVMPlugin,
                param_grid=ClassifierSVMPlugin.item_grid(C=[1.0, 10], kernel=["rbf"]),
                scoring="f1_weighted",
                cv=2,
            ),
        ],
        metrics=[F1(), Precission(), Recall()],
    )

    pipeline.fit(X, y)

    # Check if the GridSearchCVPlugin is present in the pipeline
    grid_search_plugin = next(
        (
            plugin
            for plugin in pipeline._filters
            if isinstance(plugin, GridSearchCVPlugin)
        ),
        None,
    )
    assert (
        grid_search_plugin is not None
    ), "GridSearchCVPlugin not found in the pipeline"

    print(grid_search_plugin._clf)  # type: ignore

    # Check if the grid search parameters are correctly set
    param_grid = grid_search_plugin._clf.param_grid  # type: ignore
    assert "ClassifierSVMPlugin__C" in param_grid, "C parameter not found in param_grid"
    assert param_grid["ClassifierSVMPlugin__C"] == [
        1.0,
        10,
    ], "C parameter values are incorrect"
    assert (
        "ClassifierSVMPlugin__kernel" in param_grid
    ), "kernel parameter not found in param_grid"
    assert param_grid["ClassifierSVMPlugin__kernel"] == [
        "rbf"
    ], "kernel parameter values are incorrect"

    # Check if the scoring and cv parameters are correctly set
    assert (
        grid_search_plugin._clf.scoring == "f1_weighted"
    ), "Incorrect scoring parameter"  # type: ignore
    assert grid_search_plugin._clf.cv == 2, "Incorrect cv parameter"  # type: ignore

    # Verify that the best parameters have been found
    assert hasattr(
        grid_search_plugin._clf, "best_params_"
    ), "Best parameters not found after fitting"
    assert isinstance(
        grid_search_plugin._clf.best_params_, dict
    ), "Best parameters should be a dictionary"
    assert (
        "ClassifierSVMPlugin__C" in grid_search_plugin._clf.best_params_
    ), "C parameter not found in best_params_"
    assert (
        "ClassifierSVMPlugin__kernel" in grid_search_plugin._clf.best_params_
    ), "kernel parameter not found in best_params_"
