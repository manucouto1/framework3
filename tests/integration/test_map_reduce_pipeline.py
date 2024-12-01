from cycler import V
import pytest
import typeguard
from framework3.base.base_clases import BasePipeline
from framework3.map_reduce.pyspark import PySparkMapReduce
from framework3.plugins.metrics.classification import F1
from framework3.plugins.pipelines.f3_pipeline import F3Pipeline
from tests.unit.test_combiner_pipeline import DummyFilter
import pytest
import numpy as np
from sklearn.datasets import load_iris
from framework3.base import XYData
from framework3.plugins.pipelines.map_reduce_feature_extractor_pipeline import MapReduceFeatureExtractorPipeline
from framework3.plugins.filters.classification import ClassifierSVMPlugin, KnnFilter
from framework3.plugins.filters.transformation.pca import PCAPlugin
from framework3.plugins.filters.transformation.scaler import StandardScalerPlugin
from rich import print as rprint

def test_map_reduce_combiner_pipeline_initialization():
    filters = [DummyFilter(), DummyFilter()]
    app_name = "test_app"
    master = "local[2]"
    
    pipeline = MapReduceFeatureExtractorPipeline(filters=filters, app_name=app_name, master=master)
    
    assert isinstance(pipeline, MapReduceFeatureExtractorPipeline)
    assert isinstance(pipeline, BasePipeline)
    assert len(pipeline.filters) == 2
    assert all(isinstance(f, DummyFilter) for f in pipeline.filters)
    assert isinstance(pipeline._map_reduce, PySparkMapReduce)
    assert pipeline._map_reduce.spark.conf.get("spark.app.name") == app_name
    assert pipeline._map_reduce.spark.conf.get("spark.master") == master
    pipeline.finish()

def test_map_reduce_combiner_pipeline_invalid_filters():
    invalid_filters = [1, "not a filter", {}]  # Invalid filters
    app_name = "test_app"
    
    with pytest.raises(typeguard.TypeCheckError) as excinfo:
        MapReduceFeatureExtractorPipeline(filters=invalid_filters, app_name=app_name)
    
    assert "is not an instance of framework3.base.base_clases.BaseFilter" in str(excinfo.value)




def test_map_reduce_combiner_pipeline_fit():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target # type: ignore
    
    # Create real filters
    pipeline = MapReduceFeatureExtractorPipeline(filters=[
        F3Pipeline(plugins=[StandardScalerPlugin(), PCAPlugin(n_components=2), KnnFilter()], metrics=[]),
        F3Pipeline(plugins=[StandardScalerPlugin(), PCAPlugin(n_components=1), KnnFilter()], metrics=[]),
        F3Pipeline(plugins=[StandardScalerPlugin(), PCAPlugin(n_components=2), ClassifierSVMPlugin()], metrics=[]),
        F3Pipeline(plugins=[StandardScalerPlugin(), PCAPlugin(n_components=1), ClassifierSVMPlugin()], metrics=[])
    ], app_name="test_app")

    print(pipeline)
    
    # Create XYData objects
    x = XYData(_hash="input_hash", _path="/input/path", _value=X)
    y = XYData(_hash="target_hash", _path="/target/path", _value=y)
    
    pipeline.fit(x, y)
    
    assert len(pipeline.filters) == 4
    
    # Check if fit was called on all filters
    for filter in pipeline.filters:
        assert filter._filters

        for f in filter._filters:
            assert hasattr(f, '_m_hash') # This attribute is set after fitting
            assert hasattr(f, '_m_path') # This attribute is set after fitting
            assert hasattr(f, '_m_str')  # This attribute is set after fitting
    
    # Test prediction
    result = pipeline.predict(x)
    
    assert isinstance(result, XYData)
    assert result.value is not None
    assert result.value.shape == (150, 4)  # We used 10 samples for prediction

    final_pieline = F3Pipeline(
        plugins=[
            pipeline,
            ClassifierSVMPlugin()
        ],
        metrics=[F1()]
    )


    final_pieline.fit(x, y)

    result = final_pieline.predict(x)

    rprint(final_pieline.evaluate(x, y, result))

    final_pieline.finish()