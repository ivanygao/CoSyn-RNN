# Source Directory

This directory contains the models, tasks, and datasets used by the project.

Before running an `nntp` command, these components must be registered through an entry-point file:

```bash
nntp --entrypoint=./src/entrypoint.py [command]
```

The entry-point file must exist and define a function named `register()`. This function should import and register the models and tasks used by the project.

Each task, together with its corresponding dataset, is registered using `TaskRegistry`:

```python
from nntp.datasets.TaskRegistry import TaskRegistry

TaskRegistry.register(
    # Registration arguments
)
```

Each model is registered using `ModelRegistry`:

```python
from nntp.models.ModelRegistry import ModelRegistry

ModelRegistry.register(
    # Registration arguments
)
```

A typical entry-point file has the following structure:

```python
from nntp.datasets.TaskRegistry import TaskRegistry
from nntp.models.ModelRegistry import ModelRegistry


def register():
    TaskRegistry.register(
        # Register a task and its corresponding dataset
    )

    ModelRegistry.register(
        # Register a model
    )
```

Place model implementations, task definitions, dataset implementations, and registration code in this directory.
