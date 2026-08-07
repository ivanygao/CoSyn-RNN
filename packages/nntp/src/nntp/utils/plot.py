from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections.abc import Callable


def reorder_legend_handles_row_major(handles, ncol):
    n = len(handles)
    nrow = int(np.ceil(n / ncol))
    return [
        handles[row * ncol + col]
        for col in range(ncol)
        for row in range(nrow)
        if row * ncol + col < n
    ]


def find_experiments(
    base_path: Path, get_experiment_tasks: callable = lambda x: x
) -> list[dict]:
    """
    dir strucutre:

        base_path/
            experiment_name/
                job_id/
                    seed/
                        actual_experiment/

    search dir and colllect all experiment path along with their task order.
    """
    base_path = base_path.expanduser().resolve()

    if not base_path.is_dir():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    experiments = []

    for experiment_name_path in sorted(
        (path for path in base_path.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        experiment_paths = []

        for job_path in sorted(
            (path for path in experiment_name_path.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            seed_paths = sorted(
                (path for path in job_path.iterdir() if path.is_dir()),
                key=lambda path: int(path.name),
            )

            for seed_path in seed_paths:
                actual_dirs = sorted(
                    path for path in seed_path.iterdir() if path.is_dir()
                )

                if len(actual_dirs) != 1:
                    raise ValueError(
                        f"Expected exactly one experiment directory under "
                        f"{seed_path}, but found {len(actual_dirs)}: {actual_dirs}"
                    )

                experiment_paths.append(actual_dirs[0])

        experiments.append(
            {
                "tasks": get_experiment_tasks(experiment_name_path.name),
                "paths": experiment_paths,
            }
        )

    return experiments


def load_experiments_parallel(
    experiments: list[dict],
    load_experiment: Callable,  # （tasks, path)
    max_workers: int = 8,
) -> list[dict]:
    # every element：
    # (experiment_index, path_index, path, experiment_tasks)
    load_jobs = [
        (experiment_index, path_index, path, experiment["tasks"])
        for experiment_index, experiment in enumerate(experiments)
        for path_index, path in enumerate(experiment["paths"])
    ]

    def worker(job):
        experiment_index, path_index, path, experiment_tasks = job
        data = load_experiment(experiment_tasks, path)
        return experiment_index, path_index, data

    output = [
        {"tasks": experiment["tasks"], "data": [None] * len(experiment["paths"])}
        for experiment in experiments
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for experiment_index, path_index, data in executor.map(worker, load_jobs):
            output[experiment_index]["data"][path_index] = data
    return output
