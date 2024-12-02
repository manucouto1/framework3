---
icon: material/download
---


# Installation Guide for Framework3

This guide will walk you through the process of installing Framework3 and its dependencies.

## Prerequisites

Before installing Framework3, ensure you have the following prerequisites:

1. Python 3.11 or higher
2. pip (Python package installer)
3. Git (for cloning the repository)

## Step 1: Clone the Repository

First, clone the Framework3 repository from GitHub:

```bash
git clone https://github.com/your-username/framework3.git
cd framework3
```

## Step 2: Set Up a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python projects:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv \Scripts\activate
  ```
- On macOS and Linux:
  ```bash
  source venv/bin/activate
  ```

## Step 3: Install Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages listed in the `requirements.txt` file.

## Step 4: Install Framework3

Install Framework3 in editable mode:

```bash
pip install -e .
```

This command installs the package in "editable" mode, which is useful for development.

## Step 5: Verify Installation

To verify that Framework3 is installed correctly, you can run a simple Python script:

```python
from framework3 import __version__

print(f"Framework3 version: {__version__}")
```

If this runs without errors and prints the version number, the installation was successful.

## Optional: Install Additional Dependencies

Depending on your specific use case, you might need to install additional dependencies:

- For GPU support (if using deep learning components):
  ```bash
  pip install torch torchvision torchaudio
  ```

- For distributed computing support:
  ```bash
  pip install pyspark
  ```

- For advanced visualization:
  ```bash
  pip install matplotlib seaborn
  ```

## Troubleshooting

If you encounter any issues during installation:

1. Ensure your Python version is compatible (3.8+).
2. Check that all prerequisites are installed correctly.
3. Make sure your virtual environment is activated when installing and using Framework3.
4. If you encounter any package conflicts, try creating a new virtual environment and reinstalling.

For more detailed error messages, you can use the verbose mode when installing:

```bash
pip install -v -e .
```

If problems persist, please check the project's issue tracker on GitHub or reach out to the maintainers for support.

## Updating Framework3

To update Framework3 to the latest version, navigate to the framework3 directory and run:

```bash
git pull origin main
pip install -e .
```

This will fetch the latest changes and reinstall the package.

## Next Steps

Now that you have Framework3 installed, you can start using it in your projects. Check out the [Quick Start Guide](../quick_start/index.md) for an introduction to using Framework3, or explore the [API Documentation](../api/index.md) for more detailed information on available modules and functions.