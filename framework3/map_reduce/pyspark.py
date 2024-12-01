from numpy import iterable
from pyspark.sql import SparkSession
from framework3.base import XYData
from typing import Callable, List, Any, cast

from framework3.base.base_map_reduce import MapReduceStrategy


class PySparkMapReduce(MapReduceStrategy):
    def __init__(self, app_name: str, master:str="local", num_workers:int=4):
        builder: SparkSession.Builder = cast(SparkSession.Builder, SparkSession.builder)
        self.spark:SparkSession = builder.appName(app_name) \
                                        .config("spark.master", master) \
                                        .config("spark.executor.instances", str(num_workers)) \
                                        .config("spark.cores.max", str(num_workers * 2))\
                                        .getOrCreate()
        self.sc = self.spark.sparkContext

        
    def map(self, data: Any, map_function:Callable[..., Any], numSlices:int|None=None) -> Any:
        self.rdd = self.sc.parallelize(data,numSlices=numSlices)
        self.mapped_rdd = self.rdd.map(map_function)

        # Aplicar transformaciones map
        return self.mapped_rdd

    def reduce(self, reduce_function:Callable[..., Any]) -> Any:
        reduced_rdd = self.mapped_rdd.reduceByKey(reduce_function)
        return reduced_rdd.collect()
    
    def stop(self):
        self.spark.stop()
