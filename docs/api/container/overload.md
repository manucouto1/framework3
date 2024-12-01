# Function Overloading with `overload.py`

The `overload.py` module provides function overloading capabilities for Python, enhancing the flexibility and expressiveness of function definitions in the Framework3 project.

## Overview

Function overloading allows multiple functions with the same name but different parameters to be defined. This module implements a custom solution for function overloading in Python, which natively doesn't support this feature.

## Key Components

The main component in this module is:

- `fundispatch`: A decorator that enables function overloading based on argument types.

## Usage

The `fundispatch` decorator is used to create overloaded functions. It allows you to define multiple implementations of a function with different parameter types, and the correct implementation is chosen at runtime based on the types of the arguments passed.

## Detailed Documentation

For a comprehensive look at the classes and methods provided by this module, refer to the auto-generated documentation below:

::: framework3.container.overload

## Examples

Here's a basic example of how to use the `fundispatch` decorator for function overloading:

```python

from framework3.container.overload import fundispatch

@fundispatch
def process(arg):
    raise NotImplementedError("Base process function not implemented")

@process.register(int)
def _(arg: int):
    return f"Processing integer: {arg}"

@process.register(str)
def _(arg: str):
    return f"Processing string: {arg}"

# Usage
print(process(42))       # Output: Processing integer: 42
print(process("Hello"))  # Output: Processing string: Hello

```