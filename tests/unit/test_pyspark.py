import pytest
import numpy as np
from framework3.map_reduce.pyspark import PySparkMapReduce
from framework3.base import XYData

@pytest.fixture
def pyspark_map_reduce(request):
    pyspark_mr =  PySparkMapReduce(app_name="test_app")
    def cleanup():
        pyspark_mr.spark.stop()

    request.addfinalizer(cleanup)
    return pyspark_mr


def test_pyspark_map_reduce_default_master():
    pyspark_mr = PySparkMapReduce(app_name="test_app", master="local")
    assert pyspark_mr.spark.conf.get("spark.master") == "local"
    assert pyspark_mr.spark.sparkContext.master == "local"

    pyspark_mr.spark.stop()


def test_pyspark_map_reduce_custom_master():
    custom_master = "local[2]"
    pyspark_mr = PySparkMapReduce(app_name="test_custom_app", master=custom_master)
    assert pyspark_mr.spark.conf.get("spark.master") == custom_master
    assert pyspark_mr.spark.sparkContext.master == custom_master
    assert pyspark_mr.spark.conf.get("spark.app.name") == "test_custom_app"
    pyspark_mr.spark.stop()

def test_map_parallelization(pyspark_map_reduce):
    test_data = XYData.mock([ "a",  "b",  "c"])
    map_function = lambda x: (x[0], len(x[0]))
    
    result = pyspark_map_reduce.map(test_data.value, map_function)
    
    assert result.count() == 3
    assert set(result.collect()) == {('a', 1), ('b', 1), ('c', 1)}

def test_map_function_application(pyspark_map_reduce):
    test_data = XYData.mock([(1, "apple"), (2, "banana"), (3, "cherry")])
    map_function = lambda x: (x[0], x[1].upper())
    
    result = pyspark_map_reduce.map(test_data.value, map_function)
    aux = result.collect()
    print(aux)
    assert result.count() == 3
    assert set(aux) == {(1, "APPLE"), (2, "BANANA"), (3, "CHERRY")}

def test_map_empty_input(pyspark_map_reduce):
    empty_data = XYData.mock([])
    map_function = lambda x: [(x[0], x[1])]
    
    result = pyspark_map_reduce.map(empty_data.value, map_function)
    
    assert result.count() == 0
    assert result.collect() == []

def test_reduce_function_application(pyspark_map_reduce):
    test_data = XYData.mock([("a", 1), ("b", 2), ("a", 3), ("b", 4), ("c", 5)])
    map_function = lambda x: (x[0], x[1])
    reduce_function = lambda x, y: x + y

    mapped_result = pyspark_map_reduce.map(test_data.value, map_function)
    reduced_result = pyspark_map_reduce.reduce(reduce_function)

    assert isinstance(reduced_result, list)
    assert dict(reduced_result) == {"a": 4, "b": 6, "c": 5}

def test_reduce_empty_mapped_rdd(pyspark_map_reduce):
    empty_data = XYData.mock([])
    map_function = lambda x: (x[0], x[1])
    reduce_function = lambda x, y: x + y

    mapped_result = pyspark_map_reduce.map(empty_data.value, map_function)
    reduced_result = pyspark_map_reduce.reduce(reduce_function)
    reduced_result = dict(reduced_result)
    assert isinstance(reduced_result, dict)
    assert reduced_result == {}

def test_complex_data_types(pyspark_map_reduce):
    complex_data = XYData.mock([
        (1, {"fruits": ["apple", "banana"], "count": 2}),
        (2, {"fruits": ["cherry", "date"], "count": 2}),
        (1, {"fruits": ["elderberry"], "count": 1})
    ])

    def map_function(x):
        return (x[0], x[1]["fruits"])

    def reduce_function(x, y):
        return x + y

    mapped_result = pyspark_map_reduce.map(complex_data.value, map_function)
    reduced_result = pyspark_map_reduce.reduce(reduce_function)

    assert isinstance(reduced_result, list)
    assert dict(reduced_result) == {1: ["apple", "banana", "elderberry"], 2: ["cherry", "date"]}

def test_maintain_key_value_structure(pyspark_map_reduce):
    test_data = XYData.mock([("a", 1), ("b", 2), ("c", 3), ("a", 4), ("b", 5)])
    map_function = lambda x: (x[0], x[1] * 2)
    reduce_function = lambda x, y: x + y

    mapped_result = pyspark_map_reduce.map(test_data.value, map_function)
    reduced_result = pyspark_map_reduce.reduce(reduce_function)
    reduced_result = dict(reduced_result)
    assert isinstance(reduced_result, dict)
    assert set(reduced_result.keys()) == {"a", "b", "c"}
    assert reduced_result == {"a": 10, "b": 14, "c": 6}