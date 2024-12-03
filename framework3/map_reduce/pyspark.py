from pyspark.sql import SparkSession
from typing import Callable, Any, cast

from framework3.base.base_map_reduce import MapReduceStrategy


class PySparkMapReduce(MapReduceStrategy):
    def __init__(self, app_name: str, master: str = "local", num_workers: int = 4):
        builder: SparkSession.Builder = cast(SparkSession.Builder, SparkSession.builder)
        spark: SparkSession = (
            builder.appName(app_name)
            .config("spark.master", master)
            .config("spark.executor.instances", str(num_workers))
            .config("spark.cores.max", str(num_workers * 2))
            .getOrCreate()
        )

        self.sc = spark.sparkContext

    def map(
        self, data: Any, map_function: Callable[..., Any], numSlices: int | None = None
    ) -> Any:
        self.rdd = self.sc.parallelize(data, numSlices=numSlices)
        self.mapped_rdd = self.rdd.map(map_function)

        # Aplicar transformaciones map
        return self.mapped_rdd

    def flatMap(
        self, data: Any, map_function: Callable[..., Any], numSlices: int | None = None
    ) -> Any:
        self.rdd = self.sc.parallelize(data, numSlices=numSlices)
        self.mapped_rdd = self.rdd.flatMap(map_function)

        # Aplicar transformaciones map
        return self.mapped_rdd

    def reduce(self, reduce_function: Callable[..., Any]) -> Any:
        result = self.mapped_rdd.reduce(reduce_function)
        return result

    def stop(self) -> None:
        self.sc.stop()
