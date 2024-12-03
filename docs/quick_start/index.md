---
icon: material/glasses
---

# Quick Start Guide for Framework3

This guide will help you get started with Framework3, demonstrating its basic usage and core concepts.

## 1. Installation

If you haven't installed Framework3 yet, please refer to the [Installation Guide](../installation/index.md) first.

## 2. Basic Concepts

Framework3 is built around the concept of pipelines, filters, and metrics. Here's a quick overview:

- **Pipelines**: Orchestrate the flow of data through various processing steps.
- **Filters**: Perform specific operations on data (e.g., classification, transformation).
- **Metrics**: Evaluate the performance of your models.

## 3. Creating Your First Pipeline

Let's create a simple pipeline that preprocesses data and performs classification.

```python
from framework3.plugins.pipelines.f3_pipeline import F3Pipeline
from framework3.plugins.filters.transformation.scaler import StandardScalerPlugin
from framework3.plugins.filters.classification import ClassifierSVMPlugin
from framework3.plugins.metrics.classification import F1
from framework3.base import XYData
from sklearn.datasets import load_iris

# Load sample data
iris = load_iris()
X, y = iris.data, iris.target

# Create XYData objects
x_data = XYData.mock(X)
y_data = XYData.mock(y)

# Create a pipeline
pipeline = F3Pipeline(
    plugins=[
        StandardScalerPlugin(),
        ClassifierSVMPlugin()
    ],
    metrics=[F1()]
)

# Fit the pipeline
pipeline.fit(x_data, y_data)

# Make predictions
predictions = pipeline.predict(x_data)

# Evaluate the pipeline
evaluation = pipeline.evaluate(x_data, y_data, predictions)
print("Evaluation results:", evaluation)
```

```console
____________________________________________________________________________________________________
Fitting pipeline...
****************************************************************************************************

* StandardScalerPlugin({}):

* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'}):
____________________________________________________________________________________________________
Predicting pipeline...
****************************************************************************************************

* StandardScalerPlugin({})

* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'})
Evaluating pipeline...
Evaluation results: {'F1': 0.9666366396423448}
```

## 4. Using Different Filters

Framework3 provides various filters for different tasks. Here's an example using a PCA transformation and a KNN classifier:

```python
from framework3.plugins.filters.transformation.pca import PCAPlugin
from framework3.plugins.filters.classification import KnnFilter

pipeline = F3Pipeline(
    plugins=[
        StandardScalerPlugin(),
        PCAPlugin(n_components=2),
        KnnFilter()
    ],
    metrics=[F1()]
)

pipeline.fit(x_data, y_data)
predictions = pipeline.predict(x_data)
evaluation = pipeline.evaluate(x_data, y_data, predictions)
print("Evaluation results with PCA and KNN:", evaluation)
```
```console
____________________________________________________________________________________________________
Fitting pipeline...
****************************************************************************************************

* StandardScalerPlugin({}):

* PCAPlugin({'n_components': 2}):

* KnnFilter({'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30, 'p': 2, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None}):
____________________________________________________________________________________________________
Predicting pipeline...
****************************************************************************************************

* StandardScalerPlugin({})

* PCAPlugin({'n_components': 2})

* KnnFilter({'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30, 'p': 2, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None})
Evaluating pipeline...
Evaluation results with PCA and KNN: {'F1': 0.9465811965811965}
```
## 5. Caching Intermediate Data

Some use cases can benefit greatly from caching intermediate data and reusing it for other pipeline combinations that differ in future filters. Framework3 provides a `Cached` filter wrapper to enable this functionality.

### Why Use Caching?

1. **Performance Improvement**: Caching can significantly reduce computation time for repetitive operations.
2. **Experimentation**: It allows for faster iteration when experimenting with different model configurations.
3. **Resource Efficiency**: Reduces the need to recompute expensive operations.

### How to Use Caching

Here's an example of how to use the `Cached` wrapper:

```python
from framework3.plugins.filters.cache.cached_filter import Cached
from framework3.storage import LocalStorage
from framework3.container import Container

# Set up storage
Container.storage = LocalStorage(storage_path='cache')

# Create a cached version of your filter or pipeline
cached_pipeline = Cached(
    filter=F3Pipeline(
        plugins=[
            StandardScalerPlugin(),
            PCAPlugin(n_components=2),
            ClassifierSVMPlugin()
        ],
        metrics=[F1(), Precission(), Recall()]
    ),
    cache_data=True,
    cache_filter=True,
    overwrite=False
)

# Use the cached pipeline
cached_pipeline.fit(x_data, y_data)
predictions = cached_pipeline.predict(x_data)

# Subsequent runs will use cached data if available
```
```console
         - El filtro F3Pipeline({'plugins': [StandardScalerPlugin({}), PCAPlugin({'n_components': 2}), ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'})], 'metrics':
[F1({'average': 'weighted'}), Precission({'average': 'weighted'}), Recall({'average': 'weighted'})], 'overwrite': False, 'store': False, 'log': False}) con hash
632894579e201291ada48a918108b799628cad00 No existe, se va a entrenar.
____________________________________________________________________________________________________
Fitting pipeline...
****************************************************************************************************

* StandardScalerPlugin({}):

* PCAPlugin({'n_components': 2}):

* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'}):
         - El filtro F3Pipeline({'plugins': [StandardScalerPlugin({}), PCAPlugin({'n_components': 2}), ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'})], 'metrics':
[F1({'average': 'weighted'}), Precission({'average': 'weighted'}), Recall({'average': 'weighted'})], 'overwrite': False, 'store': False, 'log': False}) Se cachea.
         * Saving in local path: /home/manuel.couto.pintos/Documents/code/framework3/cache/F3Pipeline/632894579e201291ada48a918108b799628cad00/model
         * Saved !
         - El dato XYData(_hash='a067777dda675fd67b152d9b66fe2a0a3cea1f52', _path='F3Pipeline/632894579e201291ada48a918108b799628cad00') No existe, se va a crear.
____________________________________________________________________________________________________
Predicting pipeline...
****************************************************************************************************

* StandardScalerPlugin({})

* PCAPlugin({'n_components': 2})

* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'})
         - El dato XYData(_hash='a067777dda675fd67b152d9b66fe2a0a3cea1f52', _path='F3Pipeline/632894579e201291ada48a918108b799628cad00') Se cachea.
         * Saving in local path: /home/manuel.couto.pintos/Documents/code/framework3/cache/F3Pipeline/632894579e201291ada48a918108b799628cad00/a067777dda675fd67b152d9b66fe2a0a3cea1f52
         * Saved !
{'F1': 0.9132552630700964, 'Precission': 0.9142245416833934, 'Recall': 0.9133333333333333}
```

## 6. Advanced Pipeline: Map-Reduce

Framework3 supports more complex pipeline structures, such as Map-Reduce pipelines. This type of pipeline is thought to process several pipelines concurrently and combine their output into a new feature set.


```python
from framework3.plugins.pipelines.map_reduce_feature_extractor_pipeline import MapReduceFeatureExtractorPipeline

map_reduce_pipeline = MapReduceFeatureExtractorPipeline(
    app_name='quick_start',
    filters=[
        F3Pipeline(plugins=[
            StandardScalerPlugin(),
            PCAPlugin(n_components=2),
            KnnFilter()
        ]),
        F3Pipeline(plugins=[
            StandardScalerPlugin(),
            PCAPlugin(n_components=3),
            ClassifierSVMPlugin()
        ])
    ])

map_reduce_pipeline.fit(x_data, y_data)
predictions = map_reduce_pipeline.predict(x_data)
print(predictions.value.shape)
```

```console
...
____________________
Predicting pipeline...
********************************************************************************
********************

* StandardScalerPlugin({'fit': <function StandardScalerPlugin.fit at
0x71cb785cd300>, 'predict': <function StandardScalerPlugin.predict at
0x71cb785ccfe0>})

* PCAPlugin({'n_components': 3, 'fit': <function PCAPlugin.fit at
0x71cb785cda80>, 'predict': <function PCAPlugin.predict at 0x71cb785cd6c0>})

* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale', 'fit':
<function ClassifierSVMPlugin.fit at 0x71caa99deb60>, 'predict': <function
ClassifierSVMPlugin.predict at 0x71caa99de980>})
(150, 2)
```

## 7. Stacking functionalities
Framework3's architecture defines a class structure that enables specific functionalities. One of the most advantageous characteristics is the ability to treat pipelines as if they were filters, allowing for the building of complicated pipelines and full use of the framework's features.

Now we'll look at an example of how we may use various parts of the framework to our advantage. First, we will use the framework's ability to store intermediate data, followed by the utility that allows us to generate features in parallel using PySpark, and finally, we will use pipelines as filters to train a new classifier with the new features generated in the previous step.

```python
from framework3.plugins.filters.classification import KnnFilter

Container.storage = LocalStorage(storage_path='cache')

pipeline = F3Pipeline(
    plugins=[
        MapReduceFeatureExtractorPipeline(
            app_name='quick_start',
            filters=[
                Cached(
                    filter=F3Pipeline(
                        plugins=[
                            StandardScalerPlugin(),
                            PCAPlugin(n_components=2),
                            ClassifierSVMPlugin()
                        ],
                    ),
                    cache_data=True,
                    cache_filter=True,
                    overwrite=False
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
pipeline.fit(x_data, y_data)
predictions = pipeline.predict(x_data)
print(pipeline.evaluate(x_data, y, predictions))

```

## Step-by-Step Explanation

1. **Caching Intermediate Data**
   - The first filter in the `MapReduceFeatureExtractorPipeline` is wrapped with `Cached`.
   - This allows the pipeline to store and reuse intermediate results, improving efficiency for repeated operations.
   - The cache is set to store both data (`cache_data=True`) and the fitted filter (`cache_filter=True`).

2. **Parallel Feature Generation with PySpark**
   - The `MapReduceFeatureExtractorPipeline` utilizes PySpark to process multiple feature extraction pipelines in parallel.
   - Three different feature extraction pipelines are defined:
     a. Cached pipeline with StandardScaler, PCA (2 components), and SVM classifier
     b. Pipeline with StandardScaler, PCA (3 components), and SVM classifier with RBF kernel
     c. Pipeline with StandardScaler, PCA (1 component), and SVM classifier with linear kernel
   - These pipelines will be executed in parallel, each generating a set of features.

3. **Combining Features and Final Classification**
   - The features generated by the parallel pipelines are combined.
   - The combined feature set is then passed to a KNN classifier (`KnnFilter(n_neighbors=2)`).
   - This demonstrates how pipelines can be used as filters within a larger pipeline structure.

4. **Evaluation Metrics**
   - The pipeline is configured with multiple evaluation metrics: F1 score, Precision, and Recall.
   - These metrics will be calculated automatically when the pipeline is evaluated.




## 6. Working with Metrics

You can use multiple metrics to evaluate your models:

```python
from framework3.plugins.metrics.classification import Accuracy, Precision, Recall

pipeline = F3Pipeline(
    plugins=[StandardScalerPlugin(), ClassifierSVMPlugin()],
    metrics=[F1(), Accuracy(), Precision(), Recall()]
)

pipeline.fit(x_data, y_data)
predictions = pipeline.predict(x_data)
evaluation = pipeline.evaluate(x_data, y_data, predictions)
print("Multi-metric evaluation:", evaluation)
```

## 7. Saving and Loading Models

Framework3 allows you to save and load trained models:

```python
# Save the model
pipeline.save("/path/to/save/model")

# Load the model
loaded_pipeline = F3Pipeline.load("/path/to/save/model")

# Use the loaded model
new_predictions = loaded_pipeline.predict(x_data)
```

## Next Steps

This quick start guide covers the basics of using Framework3. For more advanced usage and detailed API documentation, please refer to the following resources:

- [API Documentation](../api/index.md)
- [Examples](../examples/index.md)
- [Best Practices](../best_practices.md)

Happy coding with Framework3!
