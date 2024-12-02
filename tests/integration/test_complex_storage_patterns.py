import pickle
from typing import Any, List, cast
import numpy as np
import pytest
import pytest
from unittest.mock import MagicMock
from framework3.plugins.filters.cache.cached_filter import Cached
from framework3.base import BaseStorage
from sklearn import datasets
from framework3.base import XYData, BaseFilter
from framework3.map_reduce.pyspark import PySparkMapReduce
from framework3.plugins.filters.cache.cached_filter import Cached
from framework3.plugins.filters.transformation.pca import PCAPlugin
from framework3.plugins.filters.transformation.scaler import StandardScalerPlugin
from framework3.plugins.pipelines.f3_pipeline import F3Pipeline
from framework3.plugins.pipelines.map_reduce_feature_extractor_pipeline import MapReduceFeatureExtractorPipeline
from framework3.plugins.filters.classification.svm import ClassifierSVMPlugin
from framework3.plugins.filters.classification.knn import KnnFilter
from framework3.plugins.metrics.classification import F1, Precission, Recall
from rich import print

from framework3.plugins.storage.local_storage import LocalStorage



@pytest.fixture
def test_data():
    iris = datasets.load_iris()
    X = XYData(
        _hash='Iris X data',
        _path=f'/datasets',
        _value=iris.data # type: ignore
    )
    y = XYData(
        _hash='Iris y data',
        _path=f'/datasets',
        _value=iris.target # type: ignore
    )
    return X, y


@pytest.fixture
def mock_storage():
    return MagicMock(spec=LocalStorage)

@pytest.fixture
def simple_cached_pipelines(test_data, mock_storage):
    pipeline1 = Cached(
        F3Pipeline(plugins=[StandardScalerPlugin(), PCAPlugin(n_components=2), KnnFilter()], metrics=[]),
        cache_data=True,
        cache_filter=True,
        overwrite=False,
        storage=mock_storage
    )

    pipeline1.fit(*test_data)
    predictions1 = pipeline1.predict(test_data[0])
    return pipeline1, predictions1, mock_storage


def test_cached_pipeline_storage_interactions(simple_cached_pipelines, test_data):
    pipeline, predictions, mock_storage = simple_cached_pipelines
    
    # Comprobar que se han guardado los filtros
    for filter in pipeline.filter.plugins:
        model_hash, model_str = filter._get_model_key(test_data[0]._hash)
        mock_storage.upload_file.assert_any_call(
            content=pickle.dumps(filter),
            context=f'/root/{filter.__class__.__name__}/{model_hash}',
            filename='model.pkl'
        )
    
    # Comprobar que se han guardado los datos de entrada
    input_hash, _ = pipeline.filter._get_data_key(pipeline.filter._m_str, test_data[0]._hash)
    mock_storage.upload_file.assert_any_call(
        content=pickle.dumps(test_data[0]),
        context=f'/root/{pipeline.filter.__class__.__name__}/{input_hash}',
        filename='input.pkl'
    )
    
    # Comprobar que se han guardado los datos de salida
    output_hash, _ = pipeline.filter._get_data_key(pipeline.filter._m_str, predictions._hash)
    mock_storage.upload_file.assert_any_call(
        content=pickle.dumps(predictions),
        context=f'/root/{pipeline.filter.__class__.__name__}/{output_hash}',
        filename='output.pkl'
    )
    
    # Verificar el número total de llamadas a upload_file
    expected_calls = len(pipeline.filter.plugins) + 2  # filtros + datos de entrada + datos de salida
    assert mock_storage.upload_file.call_count == expected_calls
    
   

@pytest.fixture
def map_reduce_pipeline():
    pipeline = MapReduceFeatureExtractorPipeline(filters=[
        F3Pipeline(plugins=[StandardScalerPlugin(), PCAPlugin(n_components=1), KnnFilter()], metrics=[]),
        F3Pipeline(plugins=[StandardScalerPlugin(), PCAPlugin(n_components=2), ClassifierSVMPlugin()], metrics=[]),
        F3Pipeline(plugins=[StandardScalerPlugin(), PCAPlugin(n_components=1), ClassifierSVMPlugin()], metrics=[])
    ], app_name="test_app")

    return pipeline

@pytest.fixture
def pipelines_anidados():
    pipeline = F3Pipeline(
        plugins=[
            MapReduceFeatureExtractorPipeline(
                app_name='quick_start',
                filters=[
                    F3Pipeline(
                        plugins=[
                            StandardScalerPlugin(),
                            Cached(
                                filter=PCAPlugin(n_components=2),
                                cache_data=True,
                                cache_filter=True,
                                overwrite=False
                            ),
                            
                            ClassifierSVMPlugin()
                        ],
                        metrics=[F1(), Precission(), Recall()]
                    ),
                    
                    
                    F3Pipeline(plugins=[
                        StandardScalerPlugin(), 
                        PCAPlugin(n_components=3), 
                        ClassifierSVMPlugin(kernel='rbf'),
                    ]),
                    F3Pipeline(plugins=[
                        StandardScalerPlugin(), 
                        PCAPlugin(n_components=1), 
                        ClassifierSVMPlugin(kernel='linear')
                    ])
                ]),
            KnnFilter(n_neighbors=2)

        ],
        metrics=[F1(), Precission(), Recall()]
    )


def test_cached_filter_after_pipeline_cached(simple_cached_pipelines):

    pipeline2 = F3Pipeline(plugins=[
        StandardScalerPlugin(), 
        Cached(PCAPlugin(n_components=2), cache_data=True, cache_filter=True, overwrite=False), 
        ClassifierSVMPlugin()], metrics=[]),