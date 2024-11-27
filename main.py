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
from framework3.cache.cached_filter import Cached

from rich import print

from sklearn import datasets

from framework3.container.container import Container

pipeline = F3Pipeline(
    plugins=[
        Cached(
            filter=PCAPlugin(n_components=1),
            cache_data=False, 
            cache_filter=False,
            # overwrite=True
            # storage=LocalStorage()
        ),
        Cached(
            filter=GridSearchCVPlugin(scoring='f1_weighted', cv=2, **ClassifierSVMPlugin.item_grid(C=[1.0, 10], kernel=['rbf'])),
            cache_data=False, 
            cache_filter=False,
            # overwrite=True
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


# pipeline2 = F3Pipeline(
#     plugins=[
#         pipeline,
#     ],
#     metrics=[]
# )

# # Test data
print(pipeline)

iris = datasets.load_iris()

X = XYData(
    _hash='Iris X partial data',
    _path=Container.storage.get_root_path(),
    _value=iris.data[:,:2] # type: ignore
)
y = XYData(
    _hash='Iris y data',
    _path=Container.storage.get_root_path(),
    _value=iris.target # type: ignore
)
print(type(X))
print(type(y))

reconstructed_pipeline.fit(X, y)
prediction = reconstructed_pipeline.predict(x=X)

y_pred = XYData.mock(prediction)
evaluate = reconstructed_pipeline.evaluate(X,y, y_pred=y_pred)

print(reconstructed_pipeline)