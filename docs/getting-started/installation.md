# Installation Guide

Legion can be installed using pip or through a development environment setup. Choose the method that best suits your needs.

## **Development Environment**

If you're a contributor, check out our [Contribution Section](../contributing/setup-development.md).

## Standard Installation

### Using pip

```bash
pip install legion
```

### Using Poetry

```bash
poetry add legion
```

## Development Installation

For development purposes, we recommend setting up a proper development environment. You have two options:

### Option 1: Manual Setup

Follow our detailed [Development Environment Setup Guide](../contributing/setup-development.md#manual-installation) for step-by-step instructions.

### Option 2: Using Nix

For a fully reproducible development environment, we recommend using Nix. Follow our [Nix Setup Guide](#option-2-using-nix) for instructions.

## Verifying Installation

After installation, verify your setup:

```python
import legion
print(legion.__version__)
```

## System Requirements

- Python 3.11 or higher
- Operating System: Linux, macOS, or Windows with WSL2
- Recommended: 8GB RAM or more for running multiple agents

## Next Steps

- Follow the [Quick Start Guide](quick-start.md)
- Learn about [Basic Concepts](basic-concepts.md)
- Try the [First Agent Example](first-agent.md)
