---
icon: material/home
---

# Welcome to Framework3
## Accelerating Machine Learning Development

Framework3 is a cutting-edge platform designed to streamline and enhance the process of developing machine learning models. Our mission is to empower data scientists and machine learning engineers with a versatile and robust toolkit for efficient model creation, experimentation, and deployment.

!!! warning
    Framework3 is currently in active development. While we strive for stability, please use caution when implementing in production environments.

## Key Features

- **Modular Architecture**: Easily combine and customize components to suit your specific needs.
- **Flexible Pipelines**: Create complex, reusable workflows for your ML projects.
- **Extensible Plugin System**: Enhance functionality with filters, metrics, and storage options.
- **Distributed Processing**: Leverage MapReduce for handling large-scale data operations.
- **Comprehensive Evaluation Tools**: Assess and optimize your models with integrated metrics.

## Core Philosophy

1. **Flexibility**: Adapt to the unique requirements of each ML project.
2. **Modularity**: Mix and match components for efficient experimentation and code reuse.
3. **Transparency**: Gain clear insights at every stage of the ML process.

## Getting Started

```python
from framework3.plugins.pipelines import F3Pipeline
from framework3.plugins.filters.classification import KnnFilter
from framework3.plugins.metrics import F1, Precision, Recall

# Create a pipeline
pipeline = F3Pipeline(
    plugins=[KnnFilter()],
    metrics=[F1(), Precision(), Recall()]
)

# Fit the model and make predictions
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# Evaluate the model
evaluation = pipeline.evaluate(X_test, y_test, y_pred=predictions)
print(evaluation)
```

For more detailed information, check out our [Quick Start Guide](quick_start/index.md) or dive into the [API Documentation](api/index.md).

## Project Structure

Framework3 is organized into a comprehensive and modular structure, designed to provide maximum flexibility and extensibility. Here's a detailed overview of the project's key components:

### Core Components

- `framework3/base/`:
      - Contains fundamental abstract classes and interfaces
      - Defines the core architecture for filters, pipelines, metrics, and plugins
      - Includes base classes for data handling and storage operations

- `framework3/container/`:
      - Implements the Dependency Injection (DI) container
      - Manages the lifecycle and dependencies of various components
      - Provides a centralized registry for filters, pipelines, and plugins

- `framework3/plugins/`:
      - Houses a rich ecosystem of extensible components:
         - `filters/`: Data preprocessing, feature engineering, and model algorithms
         - `pipelines/`: Predefined and customizable ML workflows
         - `metrics/`: Various evaluation metrics for model performance
         - `storage/`: Different storage backends for data and model persistence

- `framework3/utils/`:
      - Contains utility functions and helper classes
      - Includes common operations for data manipulation, logging, and configuration

### Additional Directories

- `docs/`:
      - Comprehensive documentation, including tutorials and API references
      - Contains this index file and other markdown documentation

- `tests/`:
      - Extensive test suite ensuring reliability and correctness
      - Includes unit tests, integration tests, and end-to-end tests

- `examples/`:
      - Practical examples and use cases demonstrating Framework3's capabilities
      - Jupyter notebooks and Python scripts for hands-on learning

### Configuration and Build

- `setup.py`: Defines package metadata and dependencies for distribution
- `requirements.txt`: Lists all Python dependencies for easy installation
- `.github/`: Contains GitHub Actions workflows for CI/CD

### Development Tools

- `pre_commit/`: Hooks and configurations for maintaining code quality
- `.gitignore`: Specifies intentionally untracked files to ignore

This structure is designed to support the core philosophy of Framework3: flexibility, modularity, and transparency. It allows for easy extension and customization while maintaining a clear and organized codebase.

## Contributing

We welcome contributions! Please read our [contribution guidelines](https://github.com/manucouto1/framework3/blob/main/docs/CONTRIBUTING.md) before submitting pull requests.

## License

Framework3 is licensed under the AGPL-3.0 license. See the [LICENSE](https://github.com/manucouto1/framework3/blob/main/LICENSE) file for more details.

---

Ready to revolutionize your ML workflow? [Get started with Framework3 today!](quick_start/index.md)
