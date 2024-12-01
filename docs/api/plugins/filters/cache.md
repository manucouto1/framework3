# Cache Management

The Cache management module in Framework3 provides a powerful caching mechanism for filters, allowing you to optimize performance by storing and reusing previously computed results.

## Overview

The `Cached` filter is a wrapper that can be applied to any other filter in Framework3. It intercepts calls to the wrapped filter and checks if the result for the given input has been previously computed and stored. If so, it returns the cached result; otherwise, it computes the result, caches it, and then returns it.

## Key Features

- Transparent caching of filter results
- Automatic cache key generation based on input data and filter parameters
- Support for custom cache storage backends
- Ability to clear cache manually

## Usage

### Basic Usage

To use the `Cached` filter, simply wrap your existing filter with it:

```python
from framework3.plugins.filters.cached_filter import Cached
from framework3.plugins.filters.transformation import PCAPlugin
from framework3.container.container import Container

@Container.bind()
class CachedPCA(Cached):
    def __init__(self, n_components=2):
        super().__init__(PCAPlugin(n_components=n_components))

# Use the cached filter
cached_pca = CachedPCA(n_components=3)
result = cached_pca.transform(input_data)
```

### Clearing Cache

You can clear the cache for a specific filter instance:

```python
cached_pca.clear_cache()
```

## Advanced Usage

### Custom Cache Storage

By default, the `Cached` filter uses the storage backend configured in the Framework3 container. However, you can specify a custom storage backend:

```python
from framework3.plugins.storage import RedisStorage

custom_storage = RedisStorage(host='localhost', port=6379, db=0)
cached_pca = CachedPCA(n_components=3, storage=custom_storage)
```

### Cache Key Generation

The `Cached` filter automatically generates cache keys based on the input data and filter parameters. If you need custom key generation logic, you can override the `_get_data_key` and `_get_model_key` methods:

```python
class CustomCachedFilter(Cached):
    def _get_data_key(self, model_str: str, data_hash: str) -> Tuple[str, str]:
        # Custom key generation logic
        return custom_key, custom_key_str

    def _get_model_key(self, data_hash: str) -> Tuple[str, str]:
        # Custom model key generation logic
        return custom_model_key, custom_model_key_str
```

## Best Practices

1. **Selective Caching**: Apply caching to computationally expensive filters or those with frequent repeated inputs.
2. **Cache Invalidation**: Implement a strategy to invalidate or update cache when underlying data or models change.
3. **Monitor Cache Usage**: Keep track of cache hit rates and storage usage to optimize your caching strategy.
4. **Consider Cache Size**: Be mindful of memory usage, especially when caching large datasets or model results.

## API Reference

For a detailed look at the `Cached` filter and its methods, refer to the auto-generated documentation below:

::: framework3.plugins.filters.cache.cached_filter

## Conclusion

The `Cached` filter in Framework3 provides a powerful and flexible way to optimize your data processing pipelines through intelligent caching. By using this filter, you can significantly reduce computation time for repeated operations while maintaining the flexibility to customize caching behavior as needed.
