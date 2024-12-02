import pytest
import numpy as np
from framework3.base import XYData, BaseFilter
from framework3.plugins.pipelines.parallel.parallel_mono_pipeline import MonoPipeline
from framework3.base.base_clases import BaseMetric

class DummyFilter(BaseFilter):
    def fit(self, x, y=None):
        pass

    def predict(self, x):
        return XYData(_hash=f"pred_{x._hash}", _path=x._path, _value=np.mean(x.value * 2, axis=1))

class DummyMetric(BaseMetric):
    def __init__(self):
        super().__init__()
    def evaluate(self, x_data, y_true, y_pred):
        return np.mean(y_true.value == y_pred.value) # type: ignore

@pytest.fixture
def sample_data():
    x = XYData(_hash="input", _path="/tmp", _value=np.array([[1, 2], [3, 4], [5, 6]]))
    y = XYData(_hash="target", _path="/tmp", _value=np.array([0, 1, 0]))
    return x, y

@pytest.fixture
def combiner_pipeline():
    filters = [DummyFilter(), DummyFilter()]
    pipeline = MonoPipeline(filters=filters)
    pipeline.metrics = [DummyMetric()]
    return pipeline

def test_init(combiner_pipeline):
    assert len(combiner_pipeline.filters) == 2
    assert all(isinstance(f, DummyFilter) for f in combiner_pipeline.filters)

def test_fit(combiner_pipeline, sample_data):
    x, y = sample_data
    combiner_pipeline.fit(x, y)
    # Since DummyFilter's fit method is empty, we just check that it doesn't raise an exception

def test_predict(combiner_pipeline, sample_data):
    x, _ = sample_data
    combiner_pipeline.fit(x, None)
    result = combiner_pipeline.predict(x)
    assert isinstance(result, XYData)
    assert result.value.shape == (6,)  # 2 filtros con un ejemplo de 3 features por ejemplo
    np.testing.assert_array_equal(result.value, np.array([ 3.,  7., 11.,  3.,  7., 11.])) # type: ignore

def test_evaluate(combiner_pipeline, sample_data):
    x, y = sample_data
    y_pred = XYData(_hash="pred", _path="/tmp", _value=np.array([0, 1, 0]))
    result = combiner_pipeline.evaluate(x, y, y_pred)
    assert isinstance(result, dict)
    assert len(result) == 1
    assert "DummyMetric" in result
    assert result["DummyMetric"] == 1.0  # Perfect match

def test_combine_features():
    outputs = [
        XYData(_hash="1", _path="/tmp", _value=np.array([[1, 2], [3, 4]])),
        XYData(_hash="2", _path="/tmp", _value=np.array([[5, 6], [7, 8]]))
    ]
    result = MonoPipeline.combine_features(outputs)
    assert isinstance(result, XYData)
    assert result.value.shape == (2, 4)
    np.testing.assert_array_equal(result.value, np.array([[1, 2, 5, 6], [3, 4, 7, 8]])) # type: ignore

# def test_empty_pipeline():
#     with pytest.raises(ValueError):
#         SequentialFeatureExtractorPipeline(filters=[])

def test_different_input_shapes():
    filters = [
        DummyFilter(),
        DummyFilter()
    ]
    pipeline = MonoPipeline(filters=filters)
    x1 = XYData(_hash="input1", _path="/tmp", _value=np.array([[1, 2], [3, 4]])) # Esto representa dos filtros devolviendo un ejemplo con 2 características
    x2 = XYData(_hash="input2", _path="/tmp", _value=np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])) # Esto representa dos filtros devolviendo dos ejemplos con 3 características cada uno
    pipeline.fit(x1, None)
    result1 = pipeline.predict(x1)
    result2 = pipeline.predict(x2)
    assert x1.value.shape == (2, 2) 
    assert x2.value.shape == (2, 2, 3)
    assert result1.value.shape == (4,)
    assert result2.value.shape == (2, 6)
