"""
--------------------------------------------------------------------------------
Author:      Denis Turcu, with modifications by Ivan Gao
Affiliation: Allen Institute for Brain Science / University of Washington
Created:     2025-11-22
Last Updated: 2025-11-22
Version:     0.1.0

Task Implementations Based On:
    - Guangyu Robert Yang, et al. 2019
    - Laura N. Driscoll, et al. 2024

License:
    MIT License (from original source)

Original Source:
    https://github.com/DenisTurcu/plasticity-switch

Notes:
    This file is a cleaned and modularized re-organization of selected components
    from the original project. Redundant abstractions, PyTorch-specific modules,
    Lightning DataModules, and unused utilities have been removed to simplify
    integration into a modern JAX-based research pipeline.
--------------------------------------------------------------------------------
"""

"""Generate and save cognitive task datasets.

Supports batch processing, multiple seeds, train/validation splits, and compressed
storage. Use generate_dataset_for_task() for single tasks or generate_and_save_datasets()
for batch generation from config files.

Usage:
    >>> x, y, c_mask = generate_dataset_for_task('delaygo', 'ry', num_trials=1000)
    >>> data = load_dataset('path/to/file.pt')
"""

from typing import Tuple
import numpy as np


def getTaskGenerator(task_name, trial_generator, hyperparameters):
    class BatchGenerator:
        def __call__(
            self,
            seed: int,
            training_size: int,
            validation_size: int,
            test_size: int,
        ) -> Tuple[
            np.ndarray,  # train data
            np.ndarray,
            np.ndarray,
            np.ndarray,  # validation data
            np.ndarray,
            np.ndarray,
            np.ndarray,  # test data
            np.ndarray,
            np.ndarray,
        ]:

            train_x, train_y, train_mask = self.generate_dataset(seed, training_size)
            validation_x, validation_y, validation_mask = self.generate_dataset(seed + 1000, validation_size, False)
            test_x, test_y, test_mask = self.generate_dataset(seed + 2000, test_size, False)
            return (
                train_x,
                train_y,
                train_mask,
                validation_x,
                validation_y,
                validation_mask,
                test_x,
                test_y,
                test_mask,
            )

        def generate_dataset(self, seed: int, size: int, noise: bool = True):
            batch_size = 1

            # Setup hyperparameters with seed for reproducibility
            hp = hyperparameters.copy()
            hp["seed"] = seed
            hp["rng"] = np.random.RandomState(seed)

            num_batches = int(np.ceil(size / batch_size))

            # Collect batches
            x_trials = []
            y_trials = []
            c_mask_trials = []

            # Generate trials in batches
            for batch_idx in range(num_batches):
                current_batch_size = min(batch_size, size - batch_idx * batch_size)

                trial = trial_generator(
                    task_name,
                    hp,
                    mode="random",
                    noise_on=noise,
                    batch_size=current_batch_size,
                )

                x_trials.append(trial.x)
                y_trials.append(trial.y)
                c_mask_trials.append(trial.c_mask)

            # Pad and concatenate all batches
            x_data = self.pad_trials_to_same_length(x_trials, axis=0)
            y_data = self.pad_trials_to_same_length(y_trials, axis=0)
            c_mask_data = self.pad_trials_to_same_length(c_mask_trials, axis=0)

            return x_data, y_data, c_mask_data

        def pad_trials_to_same_length(self, trials_list: list[np.ndarray], axis: int = 0) -> np.ndarray:
            """Pad trials to match max length and concatenate.

            Args:
                trials_list: list of arrays with shape (time, batch, features)
                axis: Axis to pad (default: 0 for time)

            Returns:
                Concatenated array padded along specified axis
            """
            if not trials_list:
                return np.array([])

            # Find maximum length along the specified axis
            max_length = max(trial.shape[axis] for trial in trials_list)

            # Pad each trial to match the maximum length
            padded_trials = []
            for trial in trials_list:
                # Create padding specification for all dimensions
                pad_width = [(0, 0)] * trial.ndim
                # Only pad along the specified axis
                pad_width[axis] = (0, max_length - trial.shape[axis])
                # Pad with zeros
                padded_trial = np.pad(trial, pad_width, mode="constant", constant_values=0)
                padded_trials.append(padded_trial)

            # Concatenate along batch dimension (axis 1)
            return np.concatenate(padded_trials, axis=1)

    return BatchGenerator
