---
icon: material/download
---

# Installation Guide for Framework3

This guide will walk you through the process of installing Framework3 using pip.

## Prerequisites

Before installing Framework3, ensure you have the following prerequisites:

1. Python 3.11 or higher
2. pip (Python package installer)

## Installation

Installing Framework3 is straightforward using pip. Follow these steps:

### Step 1: Set Up a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python projects:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS and Linux:
  ```bash
  source venv/bin/activate
  ```

### Step 2: Install Framework3

Install Framework3 directly from PyPI using pip:

```bash
pip install framework3
```

This command will install the latest stable version of Framework3 and all its dependencies.

## Verify Installation

To verify that Framework3 is installed correctly, you can run a simple Python script:

```python
from framework3 import __version__

print(f"Framework3 version: {__version__}")
```

If this runs without errors and prints the version number, the installation was successful.

## Updating Framework3

To update Framework3 to the latest version, simply run:

```bash
pip install --upgrade framework3
```

## Troubleshooting

If you encounter any issues during installation:

1. Ensure your Python version is 3.11 or higher.
2. Make sure pip is up to date: `pip install --upgrade pip`
3. If you're using a virtual environment, ensure it's activated when installing and using Framework3.

For more detailed error messages, you can use the verbose mode when installing:

```bash
pip install -v framework3
```

If problems persist, please check the project's [issue tracker on GitHub](https://github.com/manucouto1/framework3/issues) or reach out to the maintainers for support.

## Next Steps

Now that you have Framework3 installed, you can start using it in your projects. Check out the [Quick Start Guide](../quick_start/index.md) for an introduction to using Framework3, or explore the [API Documentation](../api/index.md) for more detailed information on available modules and functions.
