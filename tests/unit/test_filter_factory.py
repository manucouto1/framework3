from framework3.container.factories.filter_factory import FilterFactory
from framework3.container.model.bind_model import BindGenericModel
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.base.base_clases import BaseFilter

import pytest



def test_filter_factory_attribute_error():
    factory = FilterFactory()
    with pytest.raises(AttributeError) as excinfo:
        _ = factory.non_existent_filter
    assert str(excinfo.value) == "'FilterFactory' object has no attribute 'non_existent_filter'"


def test_add_new_filter():
    factory = FilterFactory()
    
    class DummyFilter(BaseFilter):
        pass
    
    factory['new_filter'] = BindGenericModel[BaseFilter](filter=DummyFilter)
    print(factory['new_filter'])
    print(type(factory.new_filter))
    assert 'new_filter' in factory._bindings
    assert issubclass(factory.new_filter, BaseFilter)
    assert factory._bindings['new_filter'].filter == DummyFilter
    assert factory._bindings['new_filter'].manager is None
    assert factory._bindings['new_filter'].wrapper is None

def test_getattr_existing_filter():
    factory = FilterFactory()
    
    class TestFilter(BaseFilter):
        pass
    
    factory.test_filter = TestFilter
    
    assert factory.test_filter == TestFilter
    assert isinstance(factory.test_filter, type)
    assert issubclass(factory.test_filter, BaseFilter)

def test_filter_factory_iteration():
    factory = FilterFactory()
    
    class FilterA(BaseFilter):
        pass
    
    class FilterB(BaseFilter):
        pass
    
    factory.filter_a = FilterA
    factory.filter_b = FilterB
    
    items = list(factory)
    
    assert len(items) == 2
    assert ('filter_a', factory.get('filter_a')) in items
    assert ('filter_b', factory.get('filter_b')) in items
    
    for name, bind_model in items:
        assert isinstance(name, str)
        assert isinstance(bind_model, BindGenericModel)
        assert bind_model.filter in [FilterA, FilterB]

def test_contains_existing_filter():
    factory = FilterFactory()
    
    class TestFilter(BaseFilter):
        pass
    
    factory.test_filter = TestFilter
    
    assert 'test_filter' in factory

def test_contains_non_existent_filter():
    factory = FilterFactory()
    
    assert 'non_existent_filter' not in factory

def test_get_non_existent_filter_without_default():
    factory = FilterFactory()

    with pytest.raises(AttributeError) as excinfo:
        factory.get('non_existent_filter')
    assert str(excinfo.value) == "'FilterFactory' object has no attribute 'non_existent_filter'"


def test_get_non_existent_filter_with_default():
    class TestFilter(BaseFilter):
        pass
    factory = FilterFactory()
    default_value = BindGenericModel[BaseFilter](filter=TestFilter)
    result = factory.get('non_existent_filter', default=default_value)
    assert result == default_value

def test_filter_factory_initialization():
    factory = FilterFactory()
    assert hasattr(factory, '_bindings')
    assert isinstance(factory._bindings, dict)
    assert len(factory._bindings) == 0

def test_add_new_filter_with_none_manager_and_wrapper():
    factory = FilterFactory()
    
    class CustomFilter(BaseFilter):
        pass
    
    factory.custom_filter = CustomFilter
    
    assert 'custom_filter' in factory._bindings
    assert isinstance(factory._bindings['custom_filter'], BindGenericModel)
    assert factory._bindings['custom_filter'].filter == CustomFilter
    assert factory._bindings['custom_filter'].manager is None
    assert factory._bindings['custom_filter'].wrapper is None