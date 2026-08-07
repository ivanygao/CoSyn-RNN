from tqdm import tqdm
import itertools
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
import re
import json
import hashlib
import os

from .datasets import DataSetManager
from .experiment import Experiment
from .logger import Logger, PIPELINE_ROOT, DATA_ROOT
from .engine import Engine


import sys
from collections import defaultdict

JAX_ECOSYSTEM_PREFIXES = {
    "jax": [
        "jax",
        "jaxlib",
        "jaxlib.xla_extension",
        "jax._src",
    ],
    "flax": [
        "flax",
        "flax.nnx",
        "flax.linen",
    ],
    "optax": [
        "optax",
    ],
    "orbax": [
        "orbax",
        "orbax.checkpoint",
    ],
    "chex": [
        "chex",
    ],
    "etils": [
        "etils",
    ],
    "ml_dtypes": [
        "ml_dtypes",
    ],
}


def assert_jax_ecosystem_not_imported() -> bool:
    """
    Assert that no JAX-related ecosystem libraries have been imported.

    Returns True if clean.
    Raises RuntimeError if any are detected.
    """
    detected = defaultdict(list)

    for module_name in sys.modules.keys():
        for group, prefixes in JAX_ECOSYSTEM_PREFIXES.items():
            for p in prefixes:
                if module_name == p or module_name.startswith(p + "."):
                    detected[group].append(module_name)

    if detected:
        lines = ["JAX ecosystem already imported:"]
        for group, mods in detected.items():
            lines.append(f"\n[{group}]")
            for m in sorted(set(mods)):
                lines.append(f"  {m}")

        msg = "\n".join(lines)

        raise RuntimeError(msg)

    return True


class Pipeline:
    def __str__(self) -> str:
        lines = []
        lines.append(f"{'='*40}")
        lines.append(f"\tPipeline Summary ")
        lines.append(f"{'-'*40}")
        lines.append(f"Name                 :{self.name}")
        lines.append(f"Note                 :{self.note}")
        lines.append(f"Description          :{self.description}")
        lines.append(f"Seed                 :{self.seed}")
        lines.append(f"Group                 :{self.group}")
        lines.append(f"Root                 :{Logger.root}")

        lines.append(f"{'-'*40}")

        lines.append(f"Workflows: (count={len(self.workflow)})")
        for item in self.workflow:
            lines.append(str(item))
        lines.append("-" * 20)

        lines.append(f"Tasks: (count={len(self.tasks)})")
        for item in self.tasks:
            lines.append(str(item.value))
        lines.append("-" * 20)

        lines.append(f"Experiments: (count={len(self.experiments)})")
        for item in self.experiments:
            lines.append(str(item))
            lines.append("-" * 20)

        lines.append("=" * 40)
        return "\n".join(lines)

    def __init__(
        self, entrypoint: str | Path, TASKS, MODELS, cfg: BaseModel, debug: bool
    ):
        name = cfg.name
        name = name.strip().lower()
        name = re.sub(r"\s+", "_", name)  # change space to _
        name = re.sub(r"[^a-z0-9_-]", "", name)  # remove special characters
        note = cfg.note
        if cfg.note:
            note = note.strip().lower()
            note = re.sub(r"\s+", "_", note)  # change space to _
            note = re.sub(r"[^a-z0-9_-]", "", note)  # remove special characters
            name = name + "_" + note

        # loading metadata
        self.name = name
        self.note = note
        self.description = cfg.description
        self.seed = cfg.seed
        self.group = f"{name}/{os.environ.get("SLURM_JOB_ID", datetime.now().strftime('%Y%m%d-%H%M%S'))}"
        self.workflow = cfg.workflow

        Logger.set_prefix("Pipeline")
        Logger.set_root(Path(f"{PIPELINE_ROOT}/{self.group}").expanduser().resolve())
        Logger.log_yaml("metadata_config.yaml", cfg)

        # Generate Dataset
        Logger.echo("Detecting datasets:")
        # isolate non-repeating dataset name to genearte dataset
        tasks = set()
        for phase in cfg.workflow:
            tasks.update(phase.training_tasks)
            tasks.update(phase.validation_tasks)
            tasks.update(phase.test_tasks)
        self.tasks = sorted(tasks)

        with tqdm(total=len(self.tasks)) as pbar:
            for task in self.tasks:
                pbar.set_description(f"[Pipeline] Preparing Dataset: {task.value}")

                DataSetManager.generate(
                    key=task.value,
                    seeds=self.seed,
                    training_size=cfg.dataset.training_size,
                    validation_size=cfg.dataset.validation_size,
                    test_size=cfg.dataset.test_size,
                    dir_path=DATA_ROOT,
                    generator=TASKS[task.value].value.generator(),
                )

                pbar.set_postfix_str(
                    f"seeds={self.seed} | RAM {Logger.get_ram_usage()}"
                )
                pbar.update(1)

        # Generate Experiment
        experiments: list[Experiment] = []
        for model_key, model_cfg in cfg.models.items():
            model_cfg_dict = model_cfg.model_dump()
            model_cfg_dict["seed"] = self.seed
            model_sweep = self.sweep_dictionary(model_cfg_dict)
            for p in model_sweep:
                cfg_hash = hashlib.sha1(
                    json.dumps(p, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:16]
                name = f"{cfg_hash}-{note}" if note else cfg_hash
                model_cfg_clean = dict(p)
                seed = model_cfg_clean.pop("seed", None)
                experiments.append(
                    Experiment(
                        name=name,
                        seed=seed,
                        model_path=MODELS[model_key.value].value.model_path,
                        debug=debug,
                        cfg=model_cfg_clean,
                    )
                )
                experiment_cfg = cfg.model_copy()
                experiment_cfg.seed = seed
                experiment_cfg.models = {model_key: model_cfg_clean}
                Logger.log_yaml(
                    Logger.root / f"{seed}" / Path(name) / "metadata_config.yaml",
                    experiment_cfg,
                )
        self.experiments = sorted(experiments, key=lambda e: e.seed)

        # save a metadata to local file
        Logger.echo_and_log("metadata_pipeline.txt", str(self))

        self.engine = Engine(entrypoint, group=self.group)

    def run(self):
        assert_jax_ecosystem_not_imported()
        self.engine.run(self.experiments)
        self.engine.wait()

    @staticmethod
    def sweep_dictionary(parameter_dictionary):
        """
        Perform full sweep on all iterable values.
        """

        # 1. Remove None
        clean_dict = {k: v for k, v in parameter_dictionary.items() if v is not None}

        # 2. Keys & values
        keys = list(clean_dict.keys())
        values = list(clean_dict.values())

        # 3. Sweep (Cartesian product)
        combos = itertools.product(*values)

        # 4. Convert to list of dicts
        return [{k: v for k, v in zip(keys, combo)} for combo in combos]
