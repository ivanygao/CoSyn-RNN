from dataclasses import dataclass
from pathlib import Path
import numpy as np
from collections.abc import Callable


@dataclass
class DataSet:
    key: str
    seed: int
    training_size: int
    validation_size: int
    test_size: int

    train_x: np.ndarray
    train_y: np.ndarray
    train_mask: np.ndarray

    validation_x: np.ndarray
    validation_y: np.ndarray
    validation_mask: np.ndarray

    test_x: np.ndarray
    test_y: np.ndarray
    test_mask: np.ndarray

    def __str__(self) -> str:
        lines = []
        lines.append(f" - DataSet       : {self.key}")
        lines.append(f"Seed             : {self.seed}")
        lines.append(f"Training Size    : {self.training_size}")
        lines.append(f"Validation Size  : {self.validation_size}")
        lines.append(f"Test Size        : {self.test_size}")
        return "\n".join(lines)


class DataSetManager:
    @staticmethod
    def build_file_path(
        dir_path: str | Path, key: str, seed: int, tag: str, size: int
    ) -> Path:
        dir_path = Path(dir_path).expanduser().resolve()
        file_name = f"{key}-{seed}-{tag}-{size}.npy"
        return dir_path / file_name

    @staticmethod
    def generate(
        key,
        seeds,
        training_size,
        validation_size,
        test_size,
        dir_path,
        generator: Callable,
    ):
        dir_path = Path(dir_path).expanduser().resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        datasets = {}

        for seed in seeds:
            try:
                dataset = DataSetManager.load(
                    key=key,
                    seed=seed,
                    training_size=training_size,
                    validation_size=validation_size,
                    test_size=test_size,
                    dir_path=dir_path,
                )

            except FileNotFoundError:
                (
                    train_x,
                    train_y,
                    train_mask,
                    validation_x,
                    validation_y,
                    validation_mask,
                    test_x,
                    test_y,
                    test_mask,
                ) = generator(
                    seed,
                    training_size,
                    validation_size,
                    test_size,
                )

                dataset = DataSet(
                    key=key,
                    seed=seed,
                    training_size=training_size,
                    validation_size=validation_size,
                    test_size=test_size,
                    train_x=train_x,
                    train_y=train_y,
                    train_mask=train_mask,
                    validation_x=validation_x,
                    validation_y=validation_y,
                    validation_mask=validation_mask,
                    test_x=test_x,
                    test_y=test_y,
                    test_mask=test_mask,
                )

                DataSetManager.save(dataset, dir_path)
                dataset = DataSetManager.load(
                    key=key,
                    seed=seed,
                    training_size=training_size,
                    validation_size=validation_size,
                    test_size=test_size,
                    dir_path=dir_path,
                )

            datasets[seed] = dataset

        return datasets

    @staticmethod
    def load(
        key: str,
        seed: int,
        training_size: int,
        validation_size: int,
        test_size: int,
        dir_path: str | Path,
        mmap_mode: str | None = "r",
    ) -> DataSet:
        dir_path = Path(dir_path).expanduser().resolve()

        def load_array(tag: str, size: int) -> np.ndarray | np.memmap:
            file_path = DataSetManager.build_file_path(
                dir_path=dir_path, key=key, seed=seed, tag=tag, size=size
            )

            if not file_path.is_file():
                raise FileNotFoundError(f"Dataset array does not exist: {file_path}")

            return np.load(file_path, mmap_mode=mmap_mode, allow_pickle=False)

        return DataSet(
            key=key,
            seed=seed,
            training_size=training_size,
            validation_size=validation_size,
            test_size=test_size,
            train_x=load_array("train-x", training_size),
            train_y=load_array("train-y", training_size),
            train_mask=load_array("train-mask", training_size),
            validation_x=load_array("validation-x", validation_size),
            validation_y=load_array("validation-y", validation_size),
            validation_mask=load_array("validation-mask", validation_size),
            test_x=load_array("test-x", test_size),
            test_y=load_array("test-y", test_size),
            test_mask=load_array("test-mask", test_size),
        )

    @staticmethod
    def save(dataset: DataSet, dir_path: str | Path) -> None:
        dir_path = Path(dir_path).expanduser().resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        arrays = (
            ("train-x", dataset.training_size, dataset.train_x),
            ("train-y", dataset.training_size, dataset.train_y),
            ("train-mask", dataset.training_size, dataset.train_mask),
            ("validation-x", dataset.validation_size, dataset.validation_x),
            ("validation-y", dataset.validation_size, dataset.validation_y),
            ("validation-mask", dataset.validation_size, dataset.validation_mask),
            ("test-x", dataset.test_size, dataset.test_x),
            ("test-y", dataset.test_size, dataset.test_y),
            ("test-mask", dataset.test_size, dataset.test_mask),
        )

        for tag, size, array in arrays:
            file_path = DataSetManager.build_file_path(
                dir_path=dir_path,
                key=dataset.key,
                seed=dataset.seed,
                tag=tag,
                size=size,
            )

            # Copy into memory first to avoid truncating a file that may also be the backing file of a memmap.
            array_copy = np.array(array, copy=True)
            np.save(file_path, array_copy, allow_pickle=False)
