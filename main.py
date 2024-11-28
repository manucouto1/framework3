from turtle import st
from typing import cast
from framework3.base.base_clases import BaseFilter, BasePipeline, BasePlugin
from framework3.container.container import Container
from framework3.plugins.filters.clasification.svm import ClassifierSVMPlugin
from framework3.plugins.filters.grid_search.cv_grid_search import GridSearchCVPlugin
from framework3.base.base_types import XYData
from framework3.plugins.filters.transformation.pca import PCAPlugin
from framework3.plugins.metrics.classification import F1, Precission, Recall
from framework3.plugins.pipelines.pipeline import F3Pipeline
from framework3.storage.local_storage import LocalStorage
from framework3.storage.s3_storage import S3Storage
from framework3.cache.cached_filter import Cached

from rich import print

from sklearn import datasets

from framework3.container.container import Container
from dotenv import load_dotenv

import os, io, pickle

load_dotenv()

os.environ["ACCESS_KEY_ID"]
os.environ["ACCESS_KEY"]
os.environ["ENDPOINT_URL"]
os.environ["REGION_NAME"]


# storage = S3Storage(bucket='test-bucket', region_name=os.environ["REGION_NAME"], access_key_id=os.environ["ACCESS_KEY_ID"], access_key=os.environ["ACCESS_KEY"], endpoint_url=os.environ["ENDPOINT_URL"])

# iris = datasets.load_iris()
# X = XYData(
#     _hash='Iris X data',
#     _path=Container.storage.get_root_path(),
#     _value=iris.data # type: ignore
# )

# file = X.value

# if type(file) != io.BytesIO:
#     binary = pickle.dumps(file)
#     stream = io.BytesIO(binary)
# else:
#     stream = file


# storage._client.put_object(
#     Body=stream,
#     Bucket='test-bucket',
#     Key=f'datasets/{X._hash}.pkl',
# )

pipeline = F3Pipeline(
    plugins=[
        Cached(
            filter=PCAPlugin(n_components=1),
            cache_data=True, 
            cache_filter=True,
            overwrite=True,
            # storage=LocalStorage()
        ),
        Cached(
            # filter=GridSearchCVPlugin(scoring='f1_weighted', cv=2, **ClassifierSVMPlugin.item_grid(C=[1.0, 10], kernel=['rbf'])),
            filter=ClassifierSVMPlugin(C=1.0, kernel='rbf'),
            cache_data=True, 
            cache_filter=True,
            overwrite=True,
            # storage=LocalStorage()
        )
    ], 
    metrics=[
        F1(), 
        Precission(), 
        Recall()
    ]
)

print(pipeline)

dumped_pipeline = pipeline.item_dump()

print(dumped_pipeline)

reconstructed_pipeline:BasePipeline = cast(BasePipeline, BasePlugin.build_from_dump(dumped_pipeline, Container.pif))


print(reconstructed_pipeline)


pipeline2 = F3Pipeline(
    plugins=[
        pipeline,
    ],
    metrics=[]
)

# Test data
print(pipeline)

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
print(X)
print(y)


reconstructed_pipeline.fit(X, y)
prediction = reconstructed_pipeline.predict(x=X)

y_pred = XYData.mock(prediction.value)

evaluate = reconstructed_pipeline.evaluate(X,y, y_pred=y_pred)

print(reconstructed_pipeline)

print(evaluate)