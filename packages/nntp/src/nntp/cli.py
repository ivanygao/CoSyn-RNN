import argparse
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import os
import yaml
import gc

from enum import Enum
from pydantic import BaseModel
from .pipeline import Pipeline
from .logger import DATA_ROOT, PIPELINE_ROOT
from .datasets import DataSetManager


def main(
    argv: list[str] | None,
    TASKS: type[Enum],
    MODELS: type[Enum],
    RootConfig: type[BaseModel],
    entrypoint: str | Path,
):
    parser = argparse.ArgumentParser(
        description="CLI for Neural Network Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
\tnntp --entrypoint=./src/entrypoint.py info -h
\tnntp --entrypoint=./src/entrypoint.py generate --task delaygo-ry delayanti-ry delaygo-ld delayanti-ld --seed 42 --training-size 1024 --validation-size 1024 --test-size 1024 --dir-path=./runtime/data
\tnntp --entrypoint=./src/entrypoint.py run --config=configs/example.yaml --note="an example notes"
\tnntp --entrypoint=./src/entrypoint.py worker --config=configs/example.yaml
""",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ===========================
    # preparing data
    # ===========================
    models = list(MODELS.__members__.keys())
    tasks = list(TASKS.__members__.keys())

    # ===========================
    # subcommand: info
    # ===========================
    info_parser = subparsers.add_parser(
        "info",
        help="Show pipeline information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
\tnntp --entrypoint=./src/entrypoint.py info --model model1 model2
\tnntp --entrypoint=./src/entrypoint.py info --models
\tnntp --entrypoint=./src/entrypoint.py info --task task0 task1
\tnntp --entrypoint=./src/entrypoint.py info --tasks
\tnntp --entrypoint=./src/entrypoint.py info --all
""",
    )

    info_parser.add_argument(
        "--model",
        nargs="+",
        choices=models,
        help="Show model information",
    )
    info_parser.add_argument(
        "--models", action="store_true", help="Show models information"
    )
    info_parser.add_argument(
        "--task",
        nargs="+",
        choices=tasks,
        help="Show task information",
    )
    info_parser.add_argument(
        "--tasks", action="store_true", help="Show tasks information"
    )
    info_parser.add_argument("--all", action="store_true", help="Show all information")

    # ===========================
    # subcommand: generate
    # ===========================
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate dataset per task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
\tnntp --entrypoint=./src/entrypoint.py generate --task task1 task2 --seed 1 42 --training-size 1024 --validation-size 1024 --test-size 1024 --dir-path=./runtime/data
""",
    )

    generate_parser.add_argument(
        "--task",
        required=True,
        nargs="+",
        choices=tasks,
        help="Which task(s) to generate dataset for",
    )
    generate_parser.add_argument(
        "--seed",
        required=True,
        nargs="+",
        type=int,
        dest="seeds",
        help="Random seed(s) for dataset generation",
    )
    generate_parser.add_argument(
        "--training-size",
        required=True,
        type=int,
        help="Number of training trials to generate dataset for",
    )
    generate_parser.add_argument(
        "--validation-size",
        required=True,
        type=int,
        help="Number of validation trials to generate dataset for",
    )
    generate_parser.add_argument(
        "--test-size",
        required=True,
        type=int,
        help="Number of test trials to generate dataset for",
    )
    generate_parser.add_argument(
        "--dir-path",
        required=True,
        type=str,
        help="Directory path to save generated datasets",
    )

    # ===========================
    # subcommand: run
    # ===========================
    run_parser = subparsers.add_parser(
        "run",
        help="Running pipeline using config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
\tnntp --entrypoint=./src/entrypoint.py run --config=configs/example.yaml
""",
    )

    run_parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Hydra config file path",
    )

    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="redirect output to stdout/stderr",
    )

    # ===========================
    # subcommand: worker
    # ===========================
    worker_parser = subparsers.add_parser(
        "worker",
        help="Run a single experiment worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
\tnntp --entrypoint=./src/entrypoint.py worker --config=configs/example.yaml
""",
    )

    worker_parser.add_argument(
        "--id",
        required=True,
        type=int,
        help="Worker Id",
    )
    worker_parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Experiment Name",
    )
    worker_parser.add_argument(
        "--group",
        required=True,
        type=str,
        help="Pipeline Group",
    )
    worker_parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Experiment-level YAML config path",
    )

    # =====================================================
    # Parse arguments
    # =====================================================

    args, unknown = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return

    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(PIPELINE_ROOT).mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------
    # Handle "info"
    # --------------------------------------------------
    if args.command == "info":

        if not (args.model or args.models or args.task or args.tasks or args.all):
            info_parser.error("You must specify at least one of the parameter")

        if args.model:
            for model in args.model:
                print(MODELS[model].value)
        if args.all or args.models:
            for model in MODELS:
                print(f"{'='*40}")
                print("\tModel Summary")
                print(f"{'-'*40}")
                print(model.value)
                print(f"{'='*40}")

        if args.task:
            for task_name in args.task:
                print(TASKS[task_name].value)
        if args.all or args.tasks:
            for task in TASKS:
                print(f"{'='*40}")
                print("\tTasks Summary")
                print(f"{'-'*40}")
                print(task.value)
                print(f"{'='*40}")

        return

    # --------------------------------------------------
    # Handle "generate"
    # --------------------------------------------------
    if args.command == "generate":
        for task_name in args.task:
            generator = TASKS[task_name].value.generator()
            _ = DataSetManager.generate(
                task_name,
                args.seeds,
                args.training_size,
                args.validation_size,
                args.test_size,
                args.dir_path,
                generator,
            )
        return

    # --------------------------------------------------
    # Handle "run"
    # --------------------------------------------------
    if args.command == "run":
        config_path = Path(args.config).expanduser().resolve()

        if not config_path.is_file():
            run_parser.error(f"Config file does not exist: {config_path}")

        config_dir = config_path.parent
        config_name = config_path.stem

        # override configs if applicable using Hydra
        with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=unknown or [])

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Pydantic validation
        validated = RootConfig(**cfg_dict)

        # Pipeline per seed for seed consistency
        pipeline = Pipeline(entrypoint, TASKS, MODELS, validated, args.debug)
        pipeline.run()
        return

    # --------------------------------------------------
    # Handle "worker"
    # --------------------------------------------------
    if args.command == "worker":
        from .worker import Worker

        with open(args.config, "r") as f:
            cfg_dict = yaml.safe_load(f)

        cfg = OmegaConf.create(cfg_dict)

        Worker(TASKS, MODELS, args.id, args.name, args.group, cfg).run()
        return


if __name__ == "__main__":
    main()
