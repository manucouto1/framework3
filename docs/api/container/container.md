# Container Class for Dependency Injection and Inversion of Control

## Overview

The `Container` class is a cornerstone of Framework3, implementing a powerful dependency injection and inversion of control (IoC) system. It acts as a centralized registry and factory for various components, promoting loose coupling, modularity, and testability in your application architecture.

## Key Features

- **Component Registration**: Easily register filters, pipelines, metrics, storage, and plugins.
- **Dependency Injection**: Automatically manages component creation and dependency injection.
- **Inversion of Control**: Shifts component lifecycle management from application code to the container.
- **Factory Pattern**: Provides factory methods for creating instances of registered components.
- **Singleton Management**: Ensures single instances of components when needed.

## Main Components

The `Container` class includes several key attributes:

- `storage`: An instance of `BaseStorage` for handling storage operations.
- `ff`: Factory for `BaseFilter` objects.
- `pf`: Factory for `BasePipeline` objects.
- `mf`: Factory for `BaseMetric` objects.
- `sf`: Factory for `BaseStorage` objects.
- `pif`: Factory for `BasePlugin` objects.

## Usage

### Registration

Components are typically registered using the `@Container.bind()` decorator:

```python
from framework3.container import Container
from framework3.base import BaseFilter

@Container.bind()
class MyCustomFilter(BaseFilter):
    def __init__(self, param1: int, param2: str):
        self.param1 = param1
        self.param2 = param2

    def transform(self, data):
        # Implementation here
        pass
```

### Retrieval

Registered components can be retrieved using the appropriate factory attribute:

```python
# Create an instance of MyCustomFilter
my_filter = Container.ff.create('MyCustomFilter', param1=10, param2='example')

# Use the filter
result = my_filter.transform(some_data)
```

### Dependency Injection

The Container automatically handles dependency injection. If a component requires other registered components, they will be automatically injected:

```python
@Container.bind()
class ComplexPipeline(BasePipeline):
    def __init__(self, filter1: MyCustomFilter, filter2: AnotherFilter):
        self.filter1 = filter1
        self.filter2 = filter2

# Create an instance of ComplexPipeline
# The Container will automatically create and inject instances of MyCustomFilter and AnotherFilter
complex_pipeline = Container.pf.create('ComplexPipeline')
```

### Singleton Management

To ensure a single instance of a component is used throughout your application, you can use the `singleton` parameter:

```python
@Container.bind(singleton=True)
class GlobalConfig:
    def __init__(self):
        self.app_name = "MyApp"
        self.version = "1.0.0"

# This will always return the same instance
config1 = Container.pif.create('GlobalConfig')
config2 = Container.pif.create('GlobalConfig')
assert config1 is config2  # True
```

## Advanced Usage

### Custom Factories

You can create custom factories for more complex initialization scenarios:

```python
@Container.bind()
class ComplexComponent:
    @classmethod
    def create(cls, special_param):
        # Custom initialization logic
        return cls(special_param)

# Use the custom factory
complex_comp = Container.pif.create('ComplexComponent', special_param='value')
```

### Runtime Configuration

The Container allows for runtime configuration and overriding of components:

```python
# Override a component at runtime
Container.pif.register('GlobalConfig', MockConfig)

# Use the overridden component
mock_config = Container.pif.create('GlobalConfig')
```

## Best Practices

1. **Single Responsibility**: Keep your components focused and adhere to the Single Responsibility Principle.
2. **Loose Coupling**: Design your components to depend on abstractions rather than concrete implementations.
3. **Testability**: Use the Container to easily swap out real implementations with mocks or stubs in your tests.
4. **Configuration**: Use the Container to manage application-wide configuration and settings.
5. **Avoid Circular Dependencies**: Be cautious of creating circular dependencies between components.

## Detailed API Documentation

For a comprehensive look at the `Container` class and its methods, refer to the auto-generated documentation below:

::: framework3.container.container

## Conclusion

The `Container` class is a powerful tool in Framework3 that promotes clean, modular, and testable code. By centralizing component management and implementing dependency injection, it allows developers to focus on building robust and flexible applications without worrying about the intricacies of object creation and lifecycle management.
