---
icon: material/bookshelf
---

# Framework3 API Documentation

Welcome to the API documentation for Framework3. This guide provides detailed information about the modules, classes, and functions that make up the Framework3 library.

## Core Components

Framework3 is built around several core components. Here's an overview of the main modules and their purposes:

### 1. Pipelines

Pipelines are the backbone of Framework3, orchestrating the flow of data through various processing steps. We have four types of pipelines divided into three categories:

#### Sequential Pipeline
- [F3Pipeline](plugins/pipelines/sequential/f3_pipeline.md): The basic sequential pipeline structure.

#### Parallel Pipelines
- [MonoPipeline](plugins/pipelines/parallel/mono_pipeline.md): A pipeline for parallel processing of independent tasks.
- [HPCPipeline](plugins/pipelines/parallel/hpc_pipeline.md): A pipeline designed for high-performance computing environments.

#### Grid Search Pipeline
- [GridSearchPipeline](plugins/pipelines/grid/g3_pipeline.md): A pipeline for performing grid search over hyperparameters.

### 2. Filters

Filters are individual processing units that can be combined within pipelines. Framework3 provides several categories of filters:

#### Classification Filters
- [SVMClassifier](plugins/filters/classification.md#svm-classifier): Support Vector Machine classifier.
- [KNNClassifier](plugins/filters/classification.md#k-nearest-neighbors-classifier): K-Nearest Neighbors classifier.

#### Clustering Filters
- [KMeansCluster](plugins/filters/clustering.md): K-Means clustering algorithm.
- [DBSCANCluster](plugins/filters/clustering.md): Density-Based Spatial Clustering of Applications with Noise.

#### Transformation Filters
- [StandardScalerFilter](plugins/filters/transformation.md#standard-scaler): Standardize features by removing the mean and scaling to unit variance.
- [PCAFilter](plugins/filters/transformation.md#pca-principal-component-analysis): Perform Principal Component Analysis for dimensionality reduction.

#### Regression Filters
- [LinearRegression](plugins/filters/regression.md#linear-regression): Simple linear regression model.
- [RandomForestRegressor](plugins/filters/regression.md): Random Forest regression model.

#### Grid Search Filters
- [GridSearchCVFilter](plugins/filters/grid_search.md): Exhaustive search over specified parameter values for an estimator.

#### Cache Filters
- [CachedFilter](plugins/filters/cache.md): A filter that caches the results of other filters to improve performance.

Each filter category serves a specific purpose in the data processing and machine learning pipeline. You can combine these filters in various ways to create complex data processing workflows.

### 3. Metrics

Metrics are used to evaluate the performance of your models. Framework3 provides metrics in several categories:

#### Classification Metrics
- [F1Score](plugins/metrics/classification.md#f1-score): F1 score metric.
- [AccuracyScore](plugins/metrics/classification.md#accuracy-score): Accuracy score metric.
- [PrecisionScore](plugins/metrics/classification.md#precision-score): Precision score metric.
- [RecallScore](plugins/metrics/classification.md#recall-score): Recall score metric.

#### Clustering Metrics
- [SilhouetteScore](plugins/metrics/clustering.md#silhouette-score): Silhouette score for evaluating cluster quality.
- [CalinskiHarabaszScore](plugins/metrics/clustering.md): Calinski-Harabasz Index for cluster validation.

#### Coherence Metrics
- [TopicCoherence](plugins/metrics/coherence.md): Measures the semantic coherence of topics in topic modeling.
- [WordEmbeddingCoherence](plugins/metrics/coherence.md): Evaluates coherence using word embeddings.

Each metric category is designed to evaluate different aspects of model performance, allowing you to choose the most appropriate metrics for your specific machine learning tasks.

### 4. Container

The Container is a central component in Framework3 that manages the registration and retrieval of various components such as filters, pipelines, metrics, and storage.

- [Container](container/container.md): Manages the registration and retrieval of components.
- [bind](container/container.md): A decorator for binding components to the Container.

## Using the API

To use any component of Framework3, you typically need to import it from its respective module and use the Container for registration and retrieval. For example:

```python
from framework3.container import Container
from framework3.base import BaseFilter, BasePipeline, BaseMetric

@Container.bind()
class MyFilter(BaseFilter):
    # Filter implementation

@Container.bind()
class MyPipeline(BasePipeline):
    # Pipeline implementation

@Container.bind()
class MyMetric(BaseMetric):
    # Metric implementation

# Retrieving components
my_filter = Container.ff["MyFilter"]()
my_pipeline = Container.pf["MyPipeline"]()
my_metric = Container.mf["MyMetric"]()
```

## Advanced Topics

> ⚠️ **Alerta:** Under development.

<!-- - [Custom Plugin Development](advanced/custom_plugins.md): Learn how to create your own custom plugins.
- [Pipeline Optimization](advanced/pipeline_optimization.md): Techniques for optimizing pipeline performance.
- [Distributed Computing](advanced/distributed_computing.md): Using Framework3 in distributed environments. -->

## API Reference

For a complete list of all classes and functions, refer to the [Full API Reference](index.md).

## Examples

To see Framework3 in action, check out our [Examples](../examples/index.md) section, which provides practical use cases and code samples.

## Contributing to Framework3

> ⚠️ **Alerta:** Under development.

<!-- If you're interested in contributing to Framework3, please read our [Contribution Guidelines](../contributing.md) and [Code of Conduct](../code_of_conduct.md). -->

## Need Help?

> ⚠️ **Alerta:** Under development.

<!-- If you encounter any issues or have questions about using Framework3, please check our [FAQ](../faq.md) or reach out to the community through our [GitHub Issues](https://github.com/your-username/framework3/issues) page. -->