---
icon: material/glasses
---

# Quick Start Guide for LabChain

This guide will help you get started with LabChain, demonstrating its basic usage and core concepts.

## 1. Installation

Install LabChain using pip:

```bash
pip install framework3
```

## 2. Basic Concepts
* LabChain is built around:
    - **Pipelines**: Orchestrate the flow of data through processing steps.
    - **Filters**: Perform specific operations on data.
    - **Metrics**: Evaluate model performance.

## 3. Creating Your First Pipeline

Let's create a simple pipeline that preprocesses data and performs classification:

```python
from framework3 import ClassifierSVMPlugin, F3Pipeline
from framework3.plugins.filters import StandardScalerPlugin
from framework3.plugins.metrics import F1
from framework3.base import XYData
from sklearn.datasets import load_iris

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target # type: ignore

# Split the data into training and test sets

X_train, X_test, y_train, y_test = XYData("Iris", "/dataset", [])\
    .train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions and evaluate
pipeline = F3Pipeline(
    filters=[
        StandardScalerPlugin(),
        ClassifierSVMPlugin()
    ],
    metrics=[F1()]
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
evaluation = pipeline.evaluate(X_test, y_test, predictions)

```

## Next Steps

- For more advanced usage and detailed API documentation, refer to:
    - [API Documentation](../api/index.md)
    - [Examples](../examples/index.md)
    - [Best Practices](../best_practices.md)

Happy coding with LabChain!
