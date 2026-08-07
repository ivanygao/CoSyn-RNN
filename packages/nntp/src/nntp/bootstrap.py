from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_entrypoint(path: str | Path) -> ModuleType:
    entrypoint = Path(path).expanduser().resolve()

    if not entrypoint.is_file():
        raise FileNotFoundError(f"NNTP entrypoint does not exist: {entrypoint}")

    project_root = entrypoint.parent
    source_root = project_root / "src"

    for search_path in (project_root, source_root):
        if search_path.is_dir():
            resolved = str(search_path.resolve())

            if resolved not in sys.path:
                sys.path.insert(0, resolved)

    spec = importlib.util.spec_from_file_location(
        "_nntp_user_entrypoint",
        entrypoint,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load NNTP entrypoint: {entrypoint}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def main(argv: list[str] | None = None) -> None:
    bootstrap_parser = argparse.ArgumentParser(
        prog="nntp",
        add_help=False,
    )

    bootstrap_parser.add_argument(
        "--entrypoint",
        required=True,
        type=Path,
        help="Path to the user project's NNTP entrypoint file",
    )

    bootstrap_args, remaining_args = bootstrap_parser.parse_known_args(argv)

    # Phase 1: execute the user entrypoint and complete registration
    entrypoint_module = load_entrypoint(bootstrap_args.entrypoint)

    register = getattr(entrypoint_module, "register", None)

    if register is not None:
        if not callable(register):
            raise TypeError("The entrypoint attribute 'register' must be callable.")

        register()

    # Export the runtime enums after registration is complete
    from nntp.datasets import TaskRegistry
    from nntp.models import ModelRegistry
    from nntp.schema import create_root_config

    TASKS = TaskRegistry.export_registry()
    TASKS_KEY = TaskRegistry.export_registry_key()

    MODELS = ModelRegistry.export_registry()
    MODELS_KEY = ModelRegistry.export_registry_key()
    MODELS_CONFIG = ModelRegistry.export_registry_config()

    # Phase 2: delegate execution to the full CLI
    from nntp.cli import main as cli_main

    cli_main(
        argv=remaining_args,
        TASKS=TASKS,
        MODELS=MODELS,
        RootConfig=create_root_config(
            TASKS_KEY=TASKS_KEY,
            MODELS_KEY=MODELS_KEY,
            MODELS_CONFIG=MODELS_CONFIG,
        ),
        entrypoint=bootstrap_args.entrypoint,
    )


if __name__ == "__main__":
    main()
