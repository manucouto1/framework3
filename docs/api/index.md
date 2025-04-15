---
icon: material/bookshelf
---

# Framework3 API Documentation

Welcome to the API documentation for Framework3. This guide details the modules, classes, and functions that form the backbone of Framework3, enabling you to build, extend, and customize ML experimentation workflows. Use the sections below to quickly navigate to the areas you need.

---

## Table of Contents

- [Core Components](#core-components)
  - [Pipelines](#pipelines)
  - [Filters](#filters)
  - [Optimizers](#optimizers)
  - [Splitters](#splitters)
  - [Metrics](#metrics)
- [Container & Dependency Injection](#container--dependency-injection)
- [Utilities & Helpers](#utilities--helpers)
- [Using the API](#using-the-api)
- [Advanced Topics](#advanced-topics)

---

## Core Components

Framework3's core components provide the fundamental building blocks. They define the structure and behavior of filters, pipelines, and more.

### Pipelines

Pipelines orchestrate the data flow through various processing steps. They are divided by type:

- **Sequential Pipeline**
  - [F3Pipeline](plugins/pipelines/sequential/f3_pipeline.md) – The basic sequential pipeline.

- **Parallel Pipelines**
  - [MonoPipeline](plugins/pipelines/parallel/mono_pipeline.md) – For parallel processing of independent tasks.
  - [HPCPipeline](plugins/pipelines/parallel/hpc_pipeline.md) – Optimized for high-performance computing environments.

### Filters

Filters are modular processing units that can be composed together within pipelines:

#### Classification Filters
- [SVMClassifier](plugins/filters/classification.md#svm-classifier) – Support Vector Machine classifier.
- [KNNClassifier](plugins/filters/classification.md#k-nearest-neighbors-classifier) – K-Nearest Neighbors classifier.

#### Clustering Filters
- [KMeansCluster](plugins/filters/clustering.md) – K-Means clustering algorithm.
- [DBSCANCluster](plugins/filters/clustering.md) – Density-Based Spatial Clustering of Applications with Noise.

#### Transformation Filters
- [StandardScalerFilter](plugins/filters/transformation.md#standard-scaler) – Standardizes features by removing the mean and scaling to unit variance.
- [PCAFilter](plugins/filters/transformation.md#pca-principal-component-analysis) – Performs Principal Component Analysis for dimensionality reduction.

#### Regression Filters
- [LinearRegression](plugins/filters/regression.md#linear-regression) – Simple linear regression model.
- [RandomForestRegressor](plugins/filters/regression.md) – Random Forest regression model.

#### Grid Search & Caching Filters
- [GridSearchCVFilter](plugins/filters/grid_search.md) – Exhaustive search over specified parameter values.
- [CachedFilter](plugins/filters/cache.md) – Caches filter results to improve performance.

### Optimizers
Optimizers help fine-tune hyperparameters for optimal performance:

- [OptunaOptimizer](plugins/optimizers/optuna_optimizer.md) – Integrates Optuna for hyperparameter tuning.
- [WandbOptimizer](plugins/optimizers/wandb_optimizer.md) – For advanced optimization using Weights & Biases.

### Splitters
Splitters divide the dataset into folds for cross-validation and other evaluation strategies:

- [KFoldSplitter](plugins/splitters/kfold_splitter.md) – Divides the data into k-folds.
- [TimeSeriesSplitter](plugins/splitters/timeseries_splitter.md) – For time series data splitting.

### Metrics

Metrics evaluate model performance across various tasks:

#### Classification Metrics
- [F1Score](plugins/metrics/classification.md#f1-score)
- [AccuracyScore](plugins/metrics/classification.md#accuracy-score)
- [PrecisionScore](plugins/metrics/classification.md#precision-score)
- [RecallScore](plugins/metrics/classification.md#recall-score)

#### Clustering & Coherence Metrics
- [SilhouetteScore](plugins/metrics/clustering.md#silhouette-score)
- [CalinskiHarabaszScore](plugins/metrics/clustering.md)
- [TopicCoherence](plugins/metrics/coherence.md#topic-coherence)
- [WordEmbeddingCoherence](plugins/metrics/coherence.md#word-embedding-coherence)

---

## Container & Dependency Injection

The Container is central to Framework3, managing the registration and retrieval of various components.

- [Container Documentation](container/container.md) – Details on the Container class and usage.
- [bind() Decorator](container/container.md#bind) – How to register components.

---

## Utilities & Helpers

Additional utility functions and helpers that support the framework:

- [Utilities Overview](utils/overview.md) – A guide to common utility functions.
- [Miscellaneous Helpers](utils/misc.md) – Other useful routines.

---

## Using the API

To utilize any component of Framework3, import it from the respective module and register it with the Container if necessary. For example:

```python
from framework3.container import Container
from framework3.base import BaseFilter, BasePipeline, BaseMetric

@Container.bind()
class MyFilter(BaseFilter):
    # Custom filter implementation

@Container.bind()
class MyPipeline(BasePipeline):
    # Custom pipeline implementation

@Container.bind()
class MyMetric(BaseMetric):
    # Custom metric implementation

# Retrieve components
my_filter = Container.ff["MyFilter"]()
my_pipeline = Container.pf["MyPipeline"]()
my_metric = Container.mf["MyMetric"]()
```
