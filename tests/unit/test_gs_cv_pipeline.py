from typing import Optional
import pytest
import numpy as np

from sklearn.model_selection import GridSearchCV
from framework3.container.container import Container
from framework3.plugins.filters.clustering.kmeans import KMeansFilter
from framework3.plugins.filters.transformation.pca import PCAPlugin
from framework3.plugins.filters.classification.svm import ClassifierSVMPlugin
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.plugins.pipelines import GridSearchPipeline
from framework3.utils.skestimator import SkWrapper


class DummyFilter(BaseFilter):
    def __init__(self, param1: int = 0, param2: str = "test"):
        super().__init__(param1=param1, param2=param2)

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        pass

    def predict(self, x: XYData) -> XYData:
        aux = x.value[:, 0]  # type: ignore
        return XYData.mock(aux)  # type: ignore


@pytest.fixture
def sample_data():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])
    return X, y


@pytest.fixture
def dummy_filter():
    return DummyFilter


def test_sk_pipeline_wrapper_with_grid_search_cv(sample_data, dummy_filter):
    X, y = sample_data

    wrapper = SkWrapper(dummy_filter)

    param_grid = {"param1": [1, 2, 3], "param2": ["test", "test2"]}

    grid_search = GridSearchCV(wrapper, param_grid, cv=2, scoring="f1_weighted")
    grid_search.fit(X, y)

    assert isinstance(grid_search.best_estimator_, SkWrapper)
    assert grid_search.best_params_ in [
        {"param1": 1, "param2": "test"},
        {"param1": 2, "param2": "test"},
        {"param1": 3, "param2": "test"},
        {"param1": 1, "param2": "test2"},
        {"param1": 2, "param2": "test2"},
        {"param1": 3, "param2": "test2"},
    ]


def test_grid_search_cv_pipeline_with_multiple_filters():
    # Sample data
    X = np.array([[1, 2, 3], [2, 3, 1], [3, 4, 5], [5, 4, 3], [4, 5, 6], [5, 6, 7]])
    y = np.array([1, 2, 3, 4, 5, 6])
    X_data = XYData(_hash="X_data", _path="/tmp", _value=X)
    y_data = XYData(_hash="y_data", _path="/tmp", _value=y)

    # Define filters and parameter grid
    param_grid = PCAPlugin.item_grid(n_components=[2]) | ClassifierSVMPlugin.item_grid(
        C=[0.1, 1], kernel=["rbf"]
    )
    filters = [PCAPlugin, ClassifierSVMPlugin]

    # Create GridSearchCVPipeline
    grid_search = GridSearchPipeline(
        filterx=filters, param_grid=param_grid, scoring="accuracy", cv=2, metrics=[]
    )

    # Fit the pipeline
    grid_search.fit(X_data, y_data)

    # Make predictions
    X_test = XYData(_hash="X_test", _path="/tmp", _value=np.array([[3.5, 4.5, 6.4]]))
    predictions = grid_search.predict(X_test)

    # Assertions
    assert isinstance(grid_search._clf, GridSearchCV)
    assert len(grid_search._clf.cv_results_["params"]) == 2

    assert grid_search._clf.best_params_ in [
        {
            "PCAPlugin__n_components": 2,
            "ClassifierSVMPlugin__gamma": "scale",
            "ClassifierSVMPlugin__C": 0.1,
            "ClassifierSVMPlugin__kernel": "rbf",
        },
        {
            "PCAPlugin__n_components": 2,
            "ClassifierSVMPlugin__gamma": "scale",
            "ClassifierSVMPlugin__C": 1,
            "ClassifierSVMPlugin__kernel": "rbf",
        },
    ]
    assert isinstance(predictions, XYData)
    assert predictions.value.shape == (1,)
    assert predictions.value[0] in [1, 2, 3, 4, 5, 6]  # type: ignore

    # Check if evaluation returns the best score
    eval_result = grid_search.evaluate(X_data, y_data, predictions)
    assert "best_score" in eval_result
    assert isinstance(eval_result["best_score"], float)
    assert 0 <= eval_result["best_score"] <= 1


def test_grid_search_cv_pipeline_best_score_and_params():
    X = np.array([[1, 2, 3], [2, 3, 1], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
    y = np.array(range(5))
    X_data = XYData(_hash="X_data", _path="/tmp", _value=X)
    y_data = XYData(_hash="y_data", _path="/tmp", _value=y)

    # Define filters and parameter grid
    param_grid = PCAPlugin.item_grid(n_components=[2]) | ClassifierSVMPlugin.item_grid(
        C=[0.1, 1], kernel=["linear", "rbf"]
    )
    filters = [PCAPlugin, ClassifierSVMPlugin]

    # Create GridSearchCVPipeline
    grid_search = GridSearchPipeline(
        filterx=filters, param_grid=param_grid, scoring="accuracy", cv=2, metrics=[]
    )

    # Fit the pipeline
    grid_search.fit(X_data, y_data)

    # Check best score
    assert hasattr(grid_search._clf, "best_score_")
    assert isinstance(grid_search._clf.best_score_, float)
    assert 0 <= grid_search._clf.best_score_ <= 1

    # Check best parameters
    assert hasattr(grid_search._clf, "best_params_")
    assert isinstance(grid_search._clf.best_params_, dict)
    assert set(grid_search._clf.best_params_.keys()) == {
        "PCAPlugin__n_components",
        "ClassifierSVMPlugin__C",
        "ClassifierSVMPlugin__kernel",
        "ClassifierSVMPlugin__gamma",
    }

    # Evaluate and check if it returns the best score
    eval_result = grid_search.evaluate(X_data, y_data, grid_search.predict(X_data))
    assert eval_result["best_score"] == grid_search._clf.best_score_


def test_grid_search_cv_pipeline_predictions():
    # Create sample data
    X = np.array([[1, 2, 3], [2, 3, 1], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
    y = np.array([0, 1, 0, 1, 0])
    X_data = XYData(_hash="X_data", _path="/tmp", _value=X)
    y_data = XYData(_hash="y_data", _path="/tmp", _value=y)

    # Define filters using .item_grid()
    filters = [PCAPlugin, ClassifierSVMPlugin]

    param_grid = PCAPlugin.item_grid(n_components=[2]) | ClassifierSVMPlugin.item_grid(
        C=[0.1, 1], kernel=["linear", "rbf"]
    )

    # Create GridSearchCVPipeline
    grid_search = GridSearchPipeline(
        filterx=filters, param_grid=param_grid, scoring="accuracy", cv=2
    )

    # Fit the pipeline
    grid_search.fit(X_data, y_data)

    # Make predictions
    X_test = XYData(
        _hash="X_test", _path="/tmp", _value=np.array([[2, 3, 4], [4, 5, 6]])
    )
    predictions = grid_search.predict(X_test)

    # Assertions
    assert isinstance(predictions, XYData)
    assert predictions.value.shape == (2,)
    assert np.all(np.isin(predictions.value, [0, 1]))  # type: ignore

    # Check if the predictions are consistent
    assert np.array_equal(predictions.value, grid_search._clf.predict(X_test.value))  # type: ignore

    # Evaluate the model
    eval_result = grid_search.evaluate(X_data, y_data, predictions)
    assert "best_score" in eval_result
    assert isinstance(eval_result["best_score"], float)
    assert 0 <= eval_result["best_score"] <= 1


def test_grid_search_cv_pipeline_with_none_input():
    from sklearn.metrics import silhouette_score

    # Create sample data
    X = np.array([[1, 1], [2, 2], [1, 1], [2, 2], [1, 1], [2, 2], [1, 1], [2, 2]])
    X_data = XYData(_hash="X_data", _path="/tmp", _value=X)

    # Define filters using item_grid
    param_grid = PCAPlugin.item_grid(n_components=[1]) | KMeansFilter.item_grid(
        n_clusters=[2]
    )

    @Container.bind()
    class AuxPlugin(BasePlugin):
        def __init__(self, metric="euclidean"):
            super().__init__(metric=metric)
            self.metric = metric

        def __call__(self, estimator, X, y=None):
            # Asumimos que el último paso del pipeline es el clustering
            estimator.predict(X)
            return silhouette_score(X, estimator.predict(X), metric=self.metric)

    # Create GridSearchCVPipeline
    grid_search = GridSearchPipeline(
        filterx=[PCAPlugin, KMeansFilter],
        param_grid=param_grid,
        scoring=AuxPlugin(metric="euclidean"),
        cv=2,
        metrics=[],
    )

    # Test fit method with y=None
    grid_search.fit(X_data, None)

    # Make predictions
    predictions = grid_search.predict(X_data)

    # Assertions
    assert isinstance(grid_search._clf, GridSearchCV)
    assert isinstance(predictions, XYData)
    assert predictions.value.shape == (8,)

    # Test evaluate method with y_true=None
    eval_result = grid_search.evaluate(X_data, None, predictions)  # type: ignore
    assert "best_score" in eval_result
    assert isinstance(eval_result["best_score"], float)

    # Check if the best parameters are in the expected format
    assert grid_search._clf.best_params_ in [
        {"PCAPlugin__n_components": 1, "KMeansFilter__n_clusters": 2},
    ]
