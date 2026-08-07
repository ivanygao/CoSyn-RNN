import importlib
from collections import defaultdict
from collections.abc import Mapping
from typing import Dict
import numpy as np


def load_class(path: str):
    """
    path format: 'module.submodule:ClassName'
    """
    if ":" not in path:
        raise ValueError(
            f"Invalid project path '{path}', expected format 'module:ClassName'"
        )

    module_path, class_name = path.split(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}'") from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{class_name}'"
        ) from e

    return cls


def reduce_metric_list(metric_list):
    """
    list[dict[str, scalar]] -> dict[str, list[scalar]]
    """
    if not metric_list:
        return {}

    keys = metric_list[0].keys()
    out = {k: [] for k in keys}

    for item in metric_list:
        for k in keys:
            out[k].append(item[k])

    return out


def to_scalar(v):
    if hasattr(v, "item"):
        return v.item()
    return v


def reduce_batch_metrics(metric_logs):
    """
    Input:
        [
            {
                "task_key": "delaygo-ry",
                "loss": scalar,
                "task": {
                    "accuracy": scalar,
                    ...
                },
                "model": {
                    "threshold": scalar,
                    ...
                }
            },
            ...
        ]

    Output:
        {
            "model": {
                "threshold": [..],
                ...
            },
            "task": {
                "loss": {
                    "delaygo-ry": [..],
                    "fdgo-ry": [..],
                    ...
                },
                "accuracy": {
                    "delaygo-ry": [..],
                    "fdgo-ry": [..],
                    ...
                },
                ...
            }
        }
    """
    model_out = defaultdict(list)
    task_out = defaultdict(lambda: defaultdict(list))

    for log in metric_logs:
        task_key = log["task_key"]

        # -------------------------
        # task-dependent loss
        # -------------------------
        task_out["loss"][task_key].append(to_scalar(log["loss"]))

        # -------------------------
        # task-specific evaluator metrics
        # -------------------------
        for metric, value in log.get("task", {}).items():
            task_out[metric][task_key].append(to_scalar(value))

        # -------------------------
        # model/global metrics
        # -------------------------
        for metric, value in log.get("model", {}).items():
            model_out[metric].append(to_scalar(value))

    return {
        "model": dict(model_out),
        "task": {metric: dict(task_values) for metric, task_values in task_out.items()},
    }


def mean_batch_metrics(
    reduced: Dict,
) -> Dict:
    """
    Compute mean values from reduced batch metrics.

    Input:
        {
            "model": {
                metric: [v1, v2, ...],
                ...
            },
            "task": {
                metric: {
                    task_key: [v1, v2, ...],
                    ...
                },
                ...
            }
        }

    Output:
        {
            "model": {
                metric: mean(v),
                ...
            },
            "task": {
                metric: {
                    task_key: mean(v),
                    ...
                },
                ...
            },
            "task_overall": {
                metric: mean_over_tasks,
                ...
            }
        }
    """

    model_mean = {}
    task_mean = {}
    task_overall = {}

    # -------------------------
    # 1. model/global mean
    # -------------------------
    for metric, values in reduced.get("model", {}).items():
        model_mean[metric] = float(np.mean(values))

    # -------------------------
    # 2. task-wise mean
    # metric -> task_key -> mean
    # -------------------------
    for metric, task_values in reduced.get("task", {}).items():
        task_mean[metric] = {}

        per_task_means = []

        for task_key, values in task_values.items():
            mean_val = float(np.mean(values))
            task_mean[metric][task_key] = mean_val
            per_task_means.append(mean_val)

        # mean over tasks, not mean over all batches
        task_overall[metric] = float(np.mean(per_task_means))

    return {
        "model": model_mean,
        "task": task_mean,
        "task_overall": task_overall,
    }


def flatten_mean_metrics(
    metric_logs: Dict,
    label: str,
) -> Dict[str, float]:
    """
    Flatten mean metrics into a logging-friendly dict.

    Input:
        {
            "model": {
                metric: value,
                ...
            },
            "task": {
                metric: {
                    task_key: value,
                    ...
                },
                ...
            },
            "task_overall": {
                metric: value,
                ...
            }
        }

    Output:
        - model/global:
              {label}/{metric} = value

        - task overall:
              {label}/{metric} = value

        - task per-task:
              {label}/{metric}/{task_key} = value
    """
    log_dict = {}

    for metric, value in metric_logs.get("model", {}).items():
        log_dict[f"{label}/model/{metric}"] = float(value)

    for metric, value in metric_logs.get("task_overall", {}).items():
        log_dict[f"{label}/{metric}/overall"] = float(value)

    for metric, task_values in metric_logs.get("task", {}).items():
        for task_key, value in task_values.items():
            log_dict[f"{label}/{metric}/{task_key}"] = float(value)

    return log_dict


def transpose_metric_history(history: list[dict]) -> dict:
    """
    Convert an epoch-major nested dictionary:

    ```
    [
        {
            "model": {"metric_a": value, ...},
            "task": {
                "loss": {"task_1": value, ...},
                ...
            },
            "task_overall": {"loss": value, ...},
        },
        ...
    ]
    ```

    into a metric-major nested dictionary:

    ```
    {
        "model": {
            "metric_a": np.ndarray(shape=(num_epochs,)),
            ...
        },
        "task": {
            "loss": {
                "task_1": np.ndarray(shape=(num_epochs,)),
                ...
            },
            ...
        },
        "task_overall": {
            "loss": np.ndarray(shape=(num_epochs,)),
            ...
        },
    }
    ```

    Metrics missing from a particular epoch are filled with np.nan.
    """

    num_epochs = len(history)

    def collect_keys(values):
        keys = set()

        for value in values:
            if isinstance(value, Mapping):
                keys.update(value.keys())

        return sorted(keys)

    def transpose(values):
        if any(isinstance(value, Mapping) for value in values):
            output = {}

            for key in collect_keys(values):
                child_values = [
                    value.get(key, np.nan) if isinstance(value, Mapping) else np.nan
                    for value in values
                ]

                output[key] = transpose(child_values)

            return output

        normalized = [np.nan if value is None else value for value in values]

        try:
            return np.asarray(normalized, dtype=float)
        except (TypeError, ValueError):
            return np.asarray(normalized, dtype=object)

    if num_epochs == 0:
        return {}

    return transpose(history)
