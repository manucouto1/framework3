from __future__ import annotations
from typing import Tuple, Type, TypeVar, get_type_hints, Dict, Optional, Any
from typeguard import typechecked
from abc import ABC, abstractmethod
from fastapi.encoders import jsonable_encoder
from framework3.base.base_types import Float
from framework3.base.base_types import XYData
from framework3.base.base_factory import BaseFactory
import inspect
import numpy as np
import hashlib

__all__ = ["BasePlugin", "BaseFilter", "BaseMetric"]

T = TypeVar("T")


class BasePlugin(ABC):
    """
    Base class for all plugins in the framework.

    This class provides core functionality for attribute management,
    serialization, and type checking.
    """

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the BasePlugin class.

        This method applies type checking to the __init__ method and all other methods,
        and inherits type annotations from abstract methods in parent classes.
        """
        instance = super().__new__(cls)

        # Apply typechecked to the __init__ method
        init_method = cls.__init__
        if init_method is not object.__init__:
            cls.__init__ = typechecked(init_method)

        # Inherit type annotations from abstract methods
        cls.__inherit_annotations()

        # Apply typechecked to all methods defined in the class
        for attr_name, attr_value in cls.__dict__.items():
            if inspect.isfunction(attr_value) and attr_name != "__init__":
                setattr(cls, attr_name, typechecked(attr_value))

        return instance

    @classmethod
    def __inherit_annotations(cls):
        """
        Inherit type annotations from abstract methods in parent classes.
        """
        for base in cls.__bases__:
            for name, method in base.__dict__.items():
                if getattr(method, "__isabstractmethod__", False):
                    if hasattr(cls, name):
                        concrete_method = getattr(cls, name)
                        abstract_annotations = get_type_hints(method)
                        concrete_annotations = get_type_hints(concrete_method)
                        combined_annotations = {
                            **abstract_annotations,
                            **concrete_annotations,
                        }
                        setattr(
                            concrete_method, "__annotations__", combined_annotations
                        )

    def __init__(self, **kwargs):
        """
        Initialize the BasePlugin instance.

        Separates public and private attributes based on their naming.
        """
        self.__dict__["_public_attributes"] = {
            k: v for k, v in kwargs.items() if not k.startswith("_")
        }
        self.__dict__["_private_attributes"] = {
            k: v for k, v in kwargs.items() if k.startswith("_")
        }

    def __getattr__(self, name):
        """
        Custom attribute getter that checks both public and private attribute dictionaries.
        """
        if name in self.__dict__.get("_public_attributes", {}):
            return self.__dict__["_public_attributes"][name]
        elif name in self.__dict__.get("_private_attributes", {}):
            return self.__dict__["_private_attributes"][name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        """
        Custom attribute setter that separates public and private attributes.
        """
        if not hasattr(self, "_private_attributes"):
            # During initialization, attributes go directly to __dict__
            super().__setattr__(name, value)
        else:
            if name.startswith("_"):
                self.__dict__["_private_attributes"][name] = value
            else:
                self.__dict__["_public_attributes"][name] = value
            super().__setattr__(name, value)

    def __repr__(self):
        """
        String representation of the plugin, showing its class name and public attributes.
        """
        return f"{self.__class__.__name__}({self._public_attributes})"

    def model_dump(self, **kwargs):
        """
        Return a copy of the public attributes.
        """
        return self._public_attributes.copy()

    def dict(self, **kwargs):
        """
        Alias for model_dump.
        """
        return self.model_dump(**kwargs)

    def json(self, **kwargs):
        """
        Return a JSON-encodable representation of the public attributes.
        """
        return jsonable_encoder(self._public_attributes, **kwargs)

    def item_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Return a dictionary representation of the plugin, including its class name and parameters.
        """
        return {
            "clazz": self.__class__.__name__,
            "params": jsonable_encoder(
                self._public_attributes,
                custom_encoder={
                    BasePlugin: lambda v: v.item_dump(),
                    type: lambda v: {"clazz": v.__name__},
                    np.integer: lambda x: int(x),
                    np.floating: lambda x: float(x),
                },
                **kwargs,
            ),
        }

    def get_extra(self) -> Dict[str, Any]:
        """
        Return a copy of the private attributes.
        """
        return self._private_attributes.copy()

    def __getstate__(self):
        """
        Prepare the object for pickling.
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Restore the object from its pickled state.
        """
        self.__dict__.update(state)

    @classmethod
    def model_validate(cls, obj):
        """
        Validate and create an instance from a dictionary.
        """
        if isinstance(obj, dict):
            return cls(**obj)
        raise ValueError(f"Cannot validate {type(obj)}")

    def __rich_repr__(self):
        """
        Rich representation of the plugin, used by the rich library.
        """
        for key, value in self._public_attributes.items():
            yield key, value

    @staticmethod
    def build_from_dump(
        dump_dict: Dict[str, Any], factory: BaseFactory[BasePlugin]
    ) -> BasePlugin | Type[BasePlugin]:
        """
        Reconstruct a plugin instance from a dumped dictionary representation.

        This method handles nested plugin structures and uses a factory to create instances.
        """
        level_clazz: Type[BasePlugin] = factory[dump_dict["clazz"]]

        if "params" in dump_dict:
            level_params: Dict[str, Any] = {}
            for k, v in dump_dict["params"].items():
                if isinstance(v, dict):
                    if "clazz" in v:
                        level_params[k] = BasePlugin.build_from_dump(v, factory)
                    else:
                        level_params[k] = v
                elif isinstance(v, list):
                    level_params[k] = [
                        BasePlugin.build_from_dump(i, factory) for i in v
                    ]
                else:
                    level_params[k] = v
            return level_clazz(**level_params)
        else:
            return level_clazz


class BaseFilter(BasePlugin):
    """
    Base class for filter components in the framework.

    This class extends BasePlugin and provides a structure for implementing
    filter operations, including fit and predict methods. It also includes
    functionality for method wrapping and serialization.

    Example:
            ```python

            from framework3.base.base_clases import BaseFilter
            from framework3.base.base_types import XYData
            from typing import Optional

            class MyCustomFilter(BaseFilter):
                def __init__(self, param1: int = 0, param2: float = 1.0):
                    super().__init__(param1=param1, param2=param2)

                def fit(self, x: XYData, y: Optional[XYData]) -> None:
                    # Implement fitting logic here
                    pass

                def predict(self, x: XYData) -> XYData:
                    # Implement prediction logic here
                    return x  # This is a placeholder, replace with actual logic
            ```

    """

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the BaseFilter class.

        This method stores the original fit and predict methods and wraps them
        with pre-processing functionality.

        Returns:
            (BaseFilter): A new instance of the BaseFilter class.

        """
        # instance = super().__new__(cls)

        # # Store original methods
        # setattr(instance, "_original_fit", instance.fit if hasattr(instance, 'fit') else None)
        # setattr(instance, "_original_predict", instance.predict if hasattr(instance, 'predict') else None)

        # # Wrap fit and predict methods
        # if hasattr(instance, 'fit'):
        #     instance.fit = instance._pre_fit_wrapp(instance.fit)
        # if hasattr(instance, 'predict'):
        #     instance.predict = instance._pre_predict_wrapp(instance.predict)

        # return instance
        instance = super().__new__(cls)

        # Store original methods
        instance._original_fit = instance.fit if hasattr(instance, "fit") else None
        instance._original_predict = (
            instance.predict if hasattr(instance, "predict") else None
        )

        # Replace fit and predict methods
        if hasattr(instance, "fit"):
            instance.fit = instance._pre_fit_wrapp
        if hasattr(instance, "predict"):
            instance.predict = instance._pre_predict_wrapp

        return instance

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseFilter instance.

        This method sets up attributes for storing model-related information.
        """
        super().__init__(*args, **kwargs)
        self._m_hash: str
        self._m_str: str
        self._m_path: str

    def _pre_fit(self, x: XYData, y: Optional[XYData]):
        m_hash, m_str = self._get_model_key(
            data_hash=f'{x._hash}, {y._hash if y is not None else ""}'
        )
        m_path = f"{self._get_model_name()}/{m_hash}"

        self._m_hash = m_hash
        self._m_path = m_path
        self._m_str = m_str
        return m_hash, m_path, m_str

    def _pre_predict(self, x: XYData):
        if not self._m_hash or not self._m_path or not self._m_str:
            raise ValueError("Cached filter model not trained or loaded")

        d_hash, _ = self._get_data_key(self._m_str, x._hash)

        new_x = XYData(
            _hash=d_hash,
            _value=x._value,
            _path=f"{self._get_model_name()}/{self._m_hash}",
        )

        return new_x

    def _pre_fit_wrapp(self, x: XYData, y: Optional[XYData]) -> None:
        self._pre_fit(x, y)
        return self._original_fit(x, y)

    def _pre_predict_wrapp(self, x: XYData) -> XYData:
        new_x = self._pre_predict(x)
        return XYData(
            _hash=new_x._hash,
            _path=new_x._path,
            _value=self._original_predict(new_x)._value,
        )

    def __getstate__(self):
        state = super().__getstate__()
        # Ensure we're storing the original methods for serialization
        state["fit"] = self._original_fit
        state["predict"] = self._original_predict
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # Restore the wrapper methods after deserialization
        self.fit = self._pre_fit_wrapp
        self.predict = self._pre_predict_wrapp

    @abstractmethod
    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        """
        Abstract method for fitting the filter to the data.

        Args:
            x (XYData): The input data.
            y (Optional[XYData]): The target data, if applicable.
        """
        ...

    @abstractmethod
    def predict(self, x: XYData) -> XYData:
        """
        Abstract method for making predictions using the filter.

        Args:
            x (XYData): The input data.

        Returns:
            XYData: The prediction results.
        """
        ...

    def _get_model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model (class name).
        """
        return self.__class__.__name__

    def _get_model_key(self, data_hash: str) -> Tuple[str, str]:
        """
        Generate a unique key for the model based on its parameters and input data.

        Args:
            data_hash (str): A hash representing the input data.

        Returns:
            Tuple[str, str]: A tuple containing the model hash and a string representation.
        """
        model_str = f"<{self.item_dump(exclude='extra_params')}>({data_hash})"
        model_hashcode = hashlib.sha1(model_str.encode("utf-8")).hexdigest()
        return model_hashcode, model_str

    def _get_data_key(self, model_str: str, data_hash: str) -> Tuple[str, str]:
        """
        Generate a unique key for the data based on the model and input data.

        Args:
            model_str (str): A string representation of the model.
            data_hash (str): A hash representing the input data.

        Returns:
            Tuple[str, str]: A tuple containing the data hash and a string representation.
        """
        data_str = f"{model_str}.predict({data_hash})"
        data_hashcode = hashlib.sha1(data_str.encode("utf-8")).hexdigest()
        return data_hashcode, data_str


class BaseMetric(BasePlugin):
    """
    Base class for implementing metric calculations.

    This abstract class defines the interface for metric evaluation in the framework.
    Subclasses should implement the `evaluate` method to provide specific metric calculations.

    Example:
        ```python
        from framework3.base.base_clases import BaseMetric
        from framework3.base.base_types import XYData
        import numpy as np

        class MyCustomMetric(BaseMetric):
            def evaluate(self, x_data: XYData, y_true: XYData, y_pred: XYData) -> float:
                # Implement metric calculation here
                # This is a simple example that calculates mean squared error
                return np.mean((y_true.value - y_pred.value) ** 2)

        ```
    """

    @abstractmethod
    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Float | np.ndarray:
        """
        Evaluate the metric based on the provided data.

        This method should be implemented by subclasses to calculate the specific metric.

        Parameters:
        -----------
        x_data : XYData
            The input data used for the prediction.
        y_true : XYData
            The ground truth or actual values.
        y_pred : XYData
            The predicted values.

        Returns:
        --------
        Float | np.ndarray
            The calculated metric value. This can be a single float or a numpy array,
            depending on the specific metric implementation.
        """
        ...
