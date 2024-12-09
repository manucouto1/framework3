# Contributing to Framework3

Thank you for your interest in contributing to Framework3! This guide outlines the process for contributing to our project.

## Quick Start

1. Fork and clone the repository
2. Set up the development environment ([Installation Guide](installation/index.md))
3. Create a branch for your changes
4. Make your changes, following our guidelines
5. Submit a pull request

## Code of Conduct

Please read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Report any unacceptable behavior to [manuel.couto.pintos@usc.es].

## How to Contribute

### Reporting Bugs and Suggesting Enhancements

Use the issue tracker to report bugs or suggest enhancements. Provide detailed descriptions and, if possible, steps to reproduce or examples.

### Pull Requests

1. Ensure all dependencies are properly managed
2. Update documentation as necessary
3. Include comprehensive tests
4. Obtain sign-off from two developers before merging

## Development Guidelines

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use Black, isort, and flake8 for formatting and linting
- Write clear, concise docstrings in Google style

### Git Workflow

- We use a modified GitFlow:
    - `main`: latest stable release
    - `develop`: integration branch
    - Feature branches: `feature/description`
    - Hotfix branches: `hotfix/description`
    - Release branches: `release/vX.Y.Z`

Commit messages should be clear, use present tense and imperative mood.

### SOLID Principles

- Adhere to SOLID principles:
    1. Single Responsibility
    2. Open/Closed
    3. Liskov Substitution
    4. Interface Segregation
    5. Dependency Inversion

### CI/CD

Our GitHub Actions pipeline includes linting, testing, documentation building, and deployment. Ensure your changes pass all CI checks.

### Documentation and Testing

- Write docstrings for all public code elements
- Update API documentation and tutorials as needed
- Aim for 80% code coverage
- Include unit and integration tests

## Submitting Changes

1. Push to your fork
2. Create a pull request to the `develop` branch
3. Describe the changes and reference relevant issues
4. Ensure all checks pass and obtain two reviews

Thank you for contributing to Framework3!
