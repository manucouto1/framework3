from dataclasses import dataclass, field
from typing import Any, Dict, List


def dict_to_dataclass(class_name: str, data: Dict[str, Any]):
    fields = {}
    annotations = {}
    
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursivamente convertir diccionarios anidados en dataclasses
            nested_class = dict_to_dataclass(key.capitalize(), value)
            fields[key] = field(default_factory=lambda: nested_class)
            annotations[key] = nested_class
        elif isinstance(value, list):
            # Manejar listas
            if value and isinstance(value[0], dict):
                # Si la lista contiene diccionarios, convertirlos en dataclasses
                item_class = dict_to_dataclass(f"{key.capitalize()}Item", value[0])
                fields[key] = field(default_factory=lambda: [item_class(**item) for item in value])
                annotations[key] = List[item_class]
            else:
                fields[key] = field(default_factory=lambda: value)
                annotations[key] = List[type(value[0])] if value else List[Any]
        else:
            fields[key] = value
            annotations[key] = type(value)
    
    return dataclass(type(class_name, (), {
        '__annotations__': annotations,
        **fields
    }))