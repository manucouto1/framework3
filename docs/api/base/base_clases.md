Certainly! I'll create a comprehensive documentation for the base classes in framework3. Here's an updated version of the Markdown file that covers the base classes:

**File: /home/manuel.couto.pintos/Documents/code/framework3/docs/api/base/base_clases.md**

```markdown
# Base Classes from framework3

This page documents the base classes provided by the `framework3.base.base_clases` module. These classes form the foundation of the framework and are designed to be extended by specific implementations.

## Module Contents

::: framework3.base.base_clases
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Class Hierarchy

- BasePlugin
  - BaseFilter
    - BasePipeline
  - BaseMetric
  - BaseStorage

## BasePlugin

`BasePlugin` is the root class for all plugins in the framework. It provides basic functionality and structure that all plugins should adhere to.

### Key Methods:

- `__init__()`: Initialize the plugin.
- `get_name()`: Return the name of the plugin.
- `get_version()`: Return the version of the plugin.

## BaseFilter

`BaseFilter` extends `BasePlugin` and serves as the base class for all filter implementations in the framework. Filters are used to process and transform data.

### Key Methods:

- `fit(x: XYData, y: Optional[XYData]) -> None`: Train the filter on the given data.
- `predict(x: XYData) -> XYData`: Apply the filter to the input data and return the result.
- `_get_model_key(data_hash: str) -> Tuple[str, str]`: Generate a unique key for the model based on input data.
- `_get_data_key(model_str: str, data_hash: str) -> Tuple[str, str]`: Generate a unique key for the output data.

## BasePipeline

`BasePipeline` extends `BaseFilter` and provides a structure for creating sequences of filters that can be applied to data in a specific order.

### Key Methods:

- `add(filter: BaseFilter) -> None`: Add a filter to the pipeline.
- `fit(x: XYData, y: Optional[XYData]) -> None`: Train all filters in the pipeline.
- `predict(x: XYData) -> XYData`: Apply all filters in the pipeline to the input data.

## BaseMetric

`BaseMetric` extends `BasePlugin` and serves as the base class for all metric implementations. Metrics are used to evaluate the performance of filters or pipelines.

### Key Methods:

- `calculate(y_true: XYData, y_pred: XYData) -> float`: Calculate the metric value given true and predicted data.

## BaseStorage

`BaseStorage` extends `BasePlugin` and provides an interface for different storage implementations (e.g., local storage, S3 storage).

### Key Methods:

- `upload_file(file: object, file_name: str, context: str, direct_stream: bool = False) -> str`: Upload a file to storage.
- `download_file(hashcode: str, context: str) -> Any`: Download a file from storage.
- `check_if_exists(hashcode: str, context: str) -> bool`: Check if a file exists in storage.
- `delete_file(hashcode: str, context: str) -> None`: Delete a file from storage.
- `list_stored_files(context: str) -> List[Any]`: List files in a specific context.

## Usage Examples

### Creating a Custom Filter

```python
from framework3.base import BaseFilter, XYData

class MyCustomFilter(BaseFilter):
    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        # Implement training logic here
        pass

    def predict(self, x: XYData) -> XYData:
        # Implement prediction logic here
        pass
```

### Creating a Pipeline

```python
from framework3.base import BasePipeline, XYData
from my_custom_filters import FilterA, FilterB, FilterC

pipeline = BasePipeline()
pipeline.add(FilterA())
pipeline.add(FilterB())
pipeline.add(FilterC())

# Train the pipeline
pipeline.fit(train_data, train_labels)

# Make predictions
predictions = pipeline.predict(test_data)
```

### Implementing a Custom Metric

```python
from framework3.base import BaseMetric, XYData
import numpy as np

class MeanSquaredError(BaseMetric):
    def calculate(self, y_true: XYData, y_pred: XYData) -> float:
        return np.mean((y_true.value - y_pred.value) ** 2)
```

## Best Practices

1. Always inherit from the appropriate base class when creating new components.
2. Implement all required methods in your custom classes.
3. Use type hints to ensure compatibility with the framework's data structures.
4. When extending `BaseFilter`, ensure that `fit()` and `predict()` methods are properly implemented.
5. When creating custom storage solutions, inherit from `BaseStorage` and implement all required methods.

## Conclusion

The base classes in framework3 provide a solid foundation for building complex data processing pipelines. By understanding and properly extending these classes, you can create custom filters, metrics, and storage solutions that seamlessly integrate with the framework's ecosystem.
```

This documentation provides a comprehensive overview of the base classes in framework3, including their purpose, key methods, usage examples, and best practices. It should give users a clear understanding of how to effectively use and extend these base classes in their Framework3 applications.