---
icon: material/bookshelf
---

# Framework3 API Documentation

Welcome to the API documentation for Framework3. This guide provides detailed information about the modules, classes, and functions that make up the Framework3 library.

## Core Components

Framework3 is built around several core components. Here's an overview of the main modules and their purposes:

### 1. Pipelines

Pipelines are the backbone of Framework3, orchestrating the flow of data through various processing steps.

- [F3Pipeline](plugins/pipelines/f3_pipeline.md): The basic pipeline structure.
- [MapReduceCombinerPipeline](plugins/pipelines/map_reduce_pipeline.md): Advanced pipeline for parallel processing.

### 2. Filters

Filters are individual processing units that can be combined within pipelines.

#### Transformation Filters
- [StandardScalerPlugin](plugins/filters/transformation.md#standard-scaler): Standardize features by removing the mean and scaling to unit variance.
- [PCAPlugin](plugins/filters/transformation.md#pca-principal-component-analysis): Perform Principal Component Analysis for dimensionality reduction.

#### Classification Filters
- [ClassifierSVMPlugin](plugins/filters/classification.md#svm-classifier): Support Vector Machine classifier.
- [KnnFilter](plugins/filters/classification.md#k-nearest-neighbors-classifier): K-Nearest Neighbors classifier.

### 3. Metrics

Metrics are used to evaluate the performance of your models.

- [F1](plugins/metrics/classification.md#f1-score): F1 score metric.
- [Accuracy](plugins/metrics/classification.md#accuracy-score): Accuracy score metric.
- [Precision](plugins/metrics/classification.md#precision-score): Precision score metric.
- [Recall](plugins/metrics/classification.md#recall-score): Recall score metric.

### 4. Base Classes

These are the foundational classes upon which Framework3 is built.

- [XYData](base/base_types.md#base-types-used-to-move-data-through-the-framework3): Base class for handling input and target data.
- [Plugin](base/base_clases.md): Base class for all plugins (filters and metrics).

## Using the API

To use any component of Framework3, you typically need to import it from its respective module. For example:

```python
from framework3.plugins.pipelines.f3_pipeline import F3Pipeline
from framework3.plugins.filters.classification import ClassifierSVMPlugin
from framework3.plugins.metrics.classification import F1
```

Then you can use these components to build your data processing and machine learning pipelines.

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
<!-- If you're interested in contributing to Framework3, please read our [Contribution Guidelines](../contributing.md) and [Code of Conduct](../code_of_conduct.md). -->
> ⚠️ **Alerta:** Under development.

## Need Help?
> ⚠️ **Alerta:** Under development.
<!-- If you encounter any issues or have questions about using Framework3, please check our [FAQ](../faq.md) or reach out to the community through our [GitHub Issues](https://github.com/your-username/framework3/issues) page. -->

This `api/index.md` file provides a comprehensive overview of the Framework3 API, guiding users through the main components and providing links to more detailed documentation for each part. It also includes sections on how to use the API, advanced topics, and where to find additional resources.

To include this API index in your MkDocs configuration, you should update the `nav` section of your `mkdocs.yml` file. Here's how you can modify it:

**File: /home/manuel.couto.pintos/Documents/code/framework3/mkdocs.yml**
```yaml
nav:
  - Home: index.md
  - Installation: installation.md
  - Quick Start: quick_start.md
  - API Documentation:
    - Overview: api/index.md
    - Pipelines:
      - F3Pipeline: api/pipelines/f3_pipeline.md
      - MapReduceCombinerPipeline: api/pipelines/map_reduce_pipeline.md
    - Filters:
      - Transformation:
        - StandardScalerPlugin: api/filters/transformation/scaler.md
        - PCAPlugin: api/filters/transformation/pca.md
      - Classification:
        - ClassifierSVMPlugin: api/filters/classification/svm.md
        - KnnFilter: api/filters/classification/knn.md
    - Metrics:
      - Classification Metrics: api/metrics/classification.md
    - Base Classes:
      - XYData: api/base/xy_data.md
      - Plugin: api/base/plugin.md
    - Advanced Topics:
      - Custom Plugin Development: api/advanced/custom_plugins.md
      - Pipeline Optimization: api/advanced/pipeline_optimization.md
      - Distributed Computing: api/advanced/distributed_computing.md
    - Full API Reference: api/index.md
  - Examples: examples/index.md
  - Contributing: contributing.md
  - Code of Conduct: code_of_conduct.md
  - FAQ: faq.md