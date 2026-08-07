from flax import nnx
import jax
import jax.numpy as jnp
from tqdm import tqdm
import wandb
import numpy as np
import orbax.checkpoint as ocp
from pathlib import Path
import re
import hashlib

from .schema import BatchMode, Strategy
from .datasets import DataSet, Task
from .logger import Logger, PIPELINE_ROOT, DATA_ROOT
from .datasets import DataSetManager
from .utils.datatype import (
    load_class,
    reduce_batch_metrics,
    mean_batch_metrics,
    flatten_mean_metrics,
    transpose_metric_history,
)


class Worker:
    @staticmethod
    def reshape_data_to_batch_layout(
        data: np.ndarray,
        per_device_batch_size: int,
        num_devices: int,
    ) -> np.ndarray:
        """
        Input:
            (T, total_batch, X)

        Output:
            (num_batches, T, global_batch_size, X)

        This operation preserves the underlying memmap when possible.
        """
        T, total_batch, X = data.shape

        global_batch_size = per_device_batch_size * num_devices

        if total_batch % global_batch_size != 0:
            raise ValueError(
                f"Dataset batch dimension {total_batch} is not divisible by "
                f"global batch size {global_batch_size}. "
                "Padding would require allocating a new array."
            )

        num_batches = total_batch // global_batch_size

        # (T, total_batch, X)
        # -> (T, num_batches, global_batch_size, X)
        data = data.reshape(
            T,
            num_batches,
            global_batch_size,
            X,
        )

        # -> (num_batches, T, global_batch_size, X)
        return data.transpose(1, 0, 2, 3)

    def merge_tasks(self, datasets, tasks, get_X, get_Y, get_M):
        # merge all tasks into one list
        Xs, Ys, Ms, Tasks = [], [], [], []
        for task in tasks:
            dataset = datasets[task]

            X = get_X(dataset)  # (num_steps, T, B, F)
            Y = get_Y(dataset)  # (num_steps, T, B, O)
            M = get_M(dataset)  # (num_steps, T, B, O)

            num_steps = X.shape[0]

            Xs.extend(X)
            Ys.extend(Y)
            Ms.extend(M)

            Tasks.extend([self.TASKS[task].value] * num_steps)
        return Xs, Ys, Ms, Tasks

    def __str__(self) -> str:
        lines = []
        lines.append(f"{'='*40}")
        lines.append(f"\tWorker Summary ")
        lines.append(f"{'-'*40}")
        lines.append(f"Worker Id            : {self.id}")
        lines.append(f"Jax Version          : {self.jax_version}")
        lines.append(f"Backend Devices       : {self.backend_devices}")
        lines.append(f"{'-'*40}")

        lines.append(f"Experiment Name      : {self.name}")
        lines.append(f"Pipeline Group       : {self.group}")
        lines.append(f"Description          : {self.description}")
        lines.append(f"Seed                 : {self.seed}")
        lines.append(f"Root                 : {Logger.root}")

        lines.append(f"Checkpoint Interval  : {self.checkpoint_interval}")
        lines.append(f"wandb Project Name   : {self.wandb}")

        lines.append(f"Feature Size         : {self.feature_size}")
        lines.append(f"Output Size          : {self.output_size}")
        lines.append(f"Training Size        : {self.training_size}")
        lines.append(f"Validation Size      : {self.validation_size}")
        lines.append(f"Test Size            : {self.test_size}")

        lines.append(f"Batch Size           : {self.batch_size}")

        lines.append(f"Workflow             : {self.workflow}")
        lines.append(f"Total Epochs         : {self.total_epochs}")
        lines.append(f"{'-'*40}")

        lines.append(f"Model              : {self.model}")
        lines.append(f"Experiment Config    : {self.experiment_config}")
        lines.append(f"{'-'*40}")

        lines.append(f"Datasets: (count={len(self.datasets)})")
        for item in self.datasets.values():
            lines.append(str(item))

        lines.append("=" * 40)
        return "\n".join(lines)

    def __init__(self, TASKS, MODELS, id, name, group, cfg):
        self.TASKS = TASKS

        self.id = id
        self.name = name
        self.group = group
        self.description = cfg.description
        self.seed = cfg.seed
        self.rngs = nnx.Rngs(self.seed)

        Logger.set_prefix(f"Worker {id}")
        Logger.set_root(
            Path(f"{PIPELINE_ROOT}/{self.group}/{self.seed}/{self.name}")
            .expanduser()
            .resolve()
        )

        # logger
        self.wandb = cfg.logger.wandb
        self.checkpoint_interval = cfg.logger.checkpoint_interval

        # dataset
        self.training_size = cfg.dataset.training_size
        self.validation_size = cfg.dataset.validation_size
        self.test_size = cfg.dataset.test_size

        # training
        match cfg.training.batch_mode:
            case BatchMode.FIXED:
                self.batch_size = cfg.training.batch_size
            case BatchMode.AUTO:
                raise ValueError(
                    f"Not yet supported batch mode: {cfg.training.batch_mode}"
                )
            case _:
                raise ValueError(f"Unsupported batch mode: {cfg.training.batch_mode}")

        # workflow
        self.workflow = cfg.workflow
        self.total_epochs = sum(phase.epochs for phase in cfg.workflow)

        # model
        model_key, self.experiment_config = next(iter(cfg.models.items()))
        model_path = MODELS[model_key].value.model_path
        self.model = load_class(model_path)

        Logger.pickle("metadata_config.blob", self.experiment_config)

        # initiate jax related environment
        self.jax_version = getattr(jax, "__version__", None)
        devices = jax.devices()
        if not devices:
            raise RuntimeError("JAX did not detect any available devices.")

        # Current worker uses exactly one device
        self.device = devices[0]
        self.backend_devices = tuple(devices)
        self.num_devices = 1

        # Single-device layout
        self.sharding = jax.sharding.SingleDeviceSharding(self.device)

        # # initiate gpu mesh for parallel training
        # # ---- GPU / multi-device ----
        # self.sharding = jax.sharding.NamedSharding(
        #     jax.make_mesh((num_devices,), ("data",)),
        #     jax.sharding.PartitionSpec(None, "data", None),
        # )

        # Loading Dataset
        Logger.echo("Detecting datasets:")
        # isolate non-repeating dataset name to genearte dataset
        tasks = set()
        for phase in cfg.workflow:
            tasks.update(phase.training_tasks)
            tasks.update(phase.validation_tasks)
            tasks.update(phase.test_tasks)
        tasks = sorted(tasks)

        # tasks
        for task in TASKS:
            task_value = task.value
            if task_value.key not in tasks:
                continue  # skip unused tasks
            task_value.decoder = load_class(task_value.decoder)
            task_value.evaluator = load_class(task_value.evaluator)(
                self.experiment_config
            )

        # load datasets
        self.datasets: dict[str, DataSet] = {}
        pbar = tqdm(tasks)
        for task in pbar:
            pbar.set_description(
                f"[Worker {self.id}] Loading Dataset: " f"seed={self.seed} | {task}"
            )
            self.datasets[task] = DataSetManager.load(
                key=task,
                seed=self.seed,
                training_size=self.training_size,
                validation_size=self.validation_size,
                test_size=self.test_size,
                dir_path=DATA_ROOT,
                mmap_mode="r",
            )
            pbar.set_postfix_str(f"RAM {Logger.get_ram_usage()}")

        # Extract shared dataset dimensions
        sample_dataset = next(iter(self.datasets.values()))
        self.feature_size = sample_dataset.train_x.shape[-1]
        self.output_size = sample_dataset.train_y.shape[-1]

        # Preprocess data to shape:
        # [num_batches, T, global_batch_size, F]
        for task, dataset in self.datasets.items():
            for field in (
                "train_x",
                "train_y",
                "train_mask",
                "validation_x",
                "validation_y",
                "validation_mask",
                "test_x",
                "test_y",
                "test_mask",
            ):
                data = getattr(dataset, field)

                reshaped_data = self.reshape_data_to_batch_layout(
                    data=data,
                    per_device_batch_size=self.batch_size,
                    num_devices=self.num_devices,
                )

                setattr(dataset, field, reshaped_data)
        Logger.echo_and_log("metadata_worker.txt", str(self))

    def run(self):
        """
        dataset.X: [T, B, F]  (trail_length, batch_size, feature_size)
        return: [T, B, O] (trail_length, batch_size, output_size)
        """

        # initial wandb
        self.wandb and wandb.init(
            resume="allow",
            id=hashlib.sha1(f"{Logger.root}".encode("utf-8")).hexdigest()[:16],
            project=self.wandb,
            name=f"{self.name}_{self.model.__name__}",
            group=self.group,
            config={
                "seed": self.seed,
                "description": self.description,
                "training_size": self.training_size,
                "validation_size": self.validation_size,
                "test_size": self.test_size,
                "batch_size": self.batch_size,
                "workflow": self.workflow,
                "model": self.model,
                **{f"parameter/{k}": v for k, v in self.experiment_config.items()},
            },
        )

        train_step = self.model.build_training_step(self.experiment_config)
        evaluation_step = self.model.build_evaluation_step(self.experiment_config)
        plot_model_state = self.model.build_plot_model_state(self.experiment_config)

        # initialize model
        model = self.model.get_model_cls()(
            nnx.Rngs(self.seed),
            self.feature_size,
            self.output_size,
            self.experiment_config,
        )
        model_state = self.model.build_model_state(model, self.experiment_config)

        # initial orbax checkpoint saver
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.checkpoint_interval,
            save_on_steps=[self.total_epochs],
        )
        mngr = ocp.CheckpointManager(Logger.root, options=options)

        # determine is this is restoring from a break point
        latest_ckpt_step = mngr.latest_step()
        if latest_ckpt_step is not None:
            if latest_ckpt_step >= self.total_epochs:
                Logger.echo(
                    f"[Resume] Existing checkpoint detected at step={latest_ckpt_step}, but total_epochs={self.total_epochs}. Workflow already completed. Exiting."
                )
                return

            Logger.echo(
                f"[Resume] Existing checkpoint detected. Resuming experiment from checkpoint step={latest_ckpt_step}"
            )

            # ---- restore model / optimizer ----
            abstract_model = nnx.eval_shape(lambda: model_state.model)
            graphdef_model, abstract_model_state = nnx.split(abstract_model)
            abstract_optimizer = nnx.eval_shape(lambda: model_state.optimizer)
            graphdef_optimizer, abstract_optimizer_state = nnx.split(abstract_optimizer)

            # change sharding info to single device sharding
            abstract_model_state = jax.tree_util.tree_map(
                lambda x: x.update(sharding=self.sharding),
                abstract_model_state,
            )
            abstract_optimizer_state = jax.tree_util.tree_map(
                lambda x: x.update(sharding=self.sharding),
                abstract_optimizer_state,
            )

            # restore checkpoint
            restored = mngr.restore(
                latest_ckpt_step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardRestore(abstract_model_state),
                    optimizer=ocp.args.StandardRestore(abstract_optimizer_state),
                ),
            )
            model_state.model = nnx.merge(graphdef_model, restored.model)
            model_state.optimizer = nnx.merge(graphdef_optimizer, restored.optimizer)

            # locate last epoch phase
            global_epoch = latest_ckpt_step
            phase_start_index = 0
            epoch_start_index = latest_ckpt_step
            for phase_idx, phase in enumerate(self.workflow):
                if epoch_start_index >= phase.epochs:
                    epoch_start_index -= phase.epochs
                else:
                    phase_start_index = phase_idx
                    break
        else:
            Logger.echo("No checkpoint found. Saving initial (step=0) checkpoint.")
            _, mdl_state = nnx.split(model_state.model)
            _, opt_state = nnx.split(model_state.optimizer)
            mngr.save(
                0,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(mdl_state),
                    optimizer=ocp.args.StandardSave(opt_state),
                ),
            )
            global_epoch = 0
            phase_start_index = 0
            epoch_start_index = 0

            # do pre-experiment step before any training happens, so that it can log the initial state of the model
            self.wandb and wandb.log(
                plot_model_state(model_state.model, Logger.root, global_epoch),
                step=global_epoch,
            )

        # looping phases
        train_raw_metrics = []
        train_mean_metrics = []
        validation_raw_metrics = []
        validation_mean_metrics = []
        test_raw_metrics = []
        test_mean_metrics = []

        # Training the model
        Logger.start_timer()
        Logger.echo("Initialization complete. Starting the Training Process...")
        phase_bar = tqdm(
            total=len(self.workflow),
            initial=phase_start_index,
            desc=f"[Worker {self.id}] [{self.name}] Phase",
            position=0,
            dynamic_ncols=False,
            ascii=True,
        )
        for phase_idx in range(phase_start_index, len(self.workflow)):
            phase = self.workflow[phase_idx]
            phase_name = phase.name.strip().lower()
            phase_name = re.sub(r"\s+", "_", phase_name)  # change space to _
            phase_name = re.sub(
                r"[^a-z0-9_-]", "", phase_name
            )  # remove special characters

            # build datasets per phase
            train_Xs, train_Ys, train_Ms, train_Tasks = self.merge_tasks(
                self.datasets,
                phase.training_tasks,
                lambda dataset: dataset.train_x,
                lambda dataset: dataset.train_y,
                lambda dataset: dataset.train_mask,
            )
            validation_Xs, validation_Ys, validation_Ms, validation_Tasks = (
                self.merge_tasks(
                    self.datasets,
                    phase.validation_tasks,
                    lambda dataset: dataset.validation_x,
                    lambda dataset: dataset.validation_y,
                    lambda dataset: dataset.validation_mask,
                )
            )
            test_Xs, test_Ys, test_Ms, test_Tasks = self.merge_tasks(
                self.datasets,
                phase.test_tasks,
                lambda dataset: dataset.test_x,
                lambda dataset: dataset.test_y,
                lambda dataset: dataset.test_mask,
            )

            if phase.training_strategy == Strategy.SHUFFLE:
                train_rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed, phase_idx, 0])
                )
                validation_rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed, phase_idx, 1])
                )
                test_rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed, phase_idx, 2])
                )

                train_perm = train_rng.permutation(len(train_Xs))
                train_Xs = [train_Xs[i] for i in train_perm]
                train_Ys = [train_Ys[i] for i in train_perm]
                train_Ms = [train_Ms[i] for i in train_perm]
                train_Tasks = [train_Tasks[i] for i in train_perm]

                validation_perm = validation_rng.permutation(len(validation_Xs))
                validation_Xs = [validation_Xs[i] for i in validation_perm]
                validation_Ys = [validation_Ys[i] for i in validation_perm]
                validation_Ms = [validation_Ms[i] for i in validation_perm]
                validation_Tasks = [validation_Tasks[i] for i in validation_perm]

                test_perm = test_rng.permutation(len(test_Xs))
                test_Xs = [test_Xs[i] for i in test_perm]
                test_Ys = [test_Ys[i] for i in test_perm]
                test_Ms = [test_Ms[i] for i in test_perm]
                test_Tasks = [test_Tasks[i] for i in test_perm]

            # training loop for sequence of tasks
            epoch_begin = epoch_start_index if phase_idx == phase_start_index else 0
            epoch_bar = tqdm(
                total=phase.epochs,
                initial=epoch_begin,
                desc=f"[Worker {str(self.id)}] [Phase {phase_idx+1}: {phase_name}] Epoch",
                position=1,
                leave=False,
                dynamic_ncols=False,
                ascii=True,
            )
            for _ in range(epoch_begin, phase.epochs):
                global_epoch += 1
                phase_bar.set_postfix_str(f"Epoch:{global_epoch}/{self.total_epochs}")

                batch_metrics, mean_metrics, flatten_metrics = self.apply_fn_to_tasks(
                    model_state,
                    train_step,
                    train_Xs,
                    train_Ys,
                    train_Ms,
                    train_Tasks,
                    "train",
                    global_epoch,
                )

                self.wandb and wandb.log(flatten_metrics, step=global_epoch)
                train_raw_metrics.append(batch_metrics)
                train_mean_metrics.append(mean_metrics)

                is_interval_checkpoint = (
                    self.checkpoint_interval != 0
                    and global_epoch % self.checkpoint_interval == 0
                )
                is_final_checkpoint = global_epoch == self.total_epochs

                if is_interval_checkpoint or is_final_checkpoint:

                    _, mdl_state = nnx.split(model_state.model)
                    _, opt_state = nnx.split(model_state.optimizer)
                    mngr.save(
                        global_epoch,
                        args=ocp.args.Composite(
                            model=ocp.args.StandardSave(mdl_state),
                            optimizer=ocp.args.StandardSave(opt_state),
                        ),
                    )
                    Logger.echo("=" * 40)
                    Logger.echo(f"Saving Checkpoint {global_epoch}...")
                    Logger.echo("-" * 40)
                    results = plot_model_state(
                        model_state.model, Logger.root, global_epoch
                    )
                    self.wandb and wandb.log(results, step=global_epoch)
                    Logger.echo("=" * 40)

                # validation loop for sequence of tasks
                batch_metrics, mean_metrics, flatten_metrics = self.apply_fn_to_tasks(
                    model_state,
                    evaluation_step,
                    validation_Xs,
                    validation_Ys,
                    validation_Ms,
                    validation_Tasks,
                    "validation",
                    global_epoch,
                )
                epoch_bar.set_postfix(
                    validation_loss=f"{flatten_metrics["validation/loss/overall"]:.6f}"
                )
                self.wandb and wandb.log(flatten_metrics, step=global_epoch)
                validation_raw_metrics.append(batch_metrics)
                validation_mean_metrics.append(mean_metrics)
                epoch_bar.update(1)

            # test loop for sequence of tasks
            batch_metrics, mean_metrics, flatten_metrics = self.apply_fn_to_tasks(
                model_state,
                evaluation_step,
                test_Xs,
                test_Ys,
                test_Ms,
                test_Tasks,
                "test",
                global_epoch,
            )
            self.wandb and wandb.log(flatten_metrics, step=global_epoch)
            Logger.echo("executing the post test evaluation")
            test_raw_metrics.append(batch_metrics)
            test_mean_metrics.append(mean_metrics)
            phase_bar.update(1)

        # clean up
        Logger.echo("=" * 40)
        Logger.echo("Training process complete. Starting the Cleaning Process...")
        mngr.wait_until_finished()
        Logger.pickle("metadata_train_raw_metrics.blob", train_raw_metrics)
        Logger.pickle(
            "metadata_train_mean_metrics.blob",
            transpose_metric_history(train_mean_metrics),
        )
        Logger.pickle("metadata_validation_raw_metrics.blob", validation_raw_metrics)
        Logger.pickle(
            "metadata_validation_mean_metrics.blob",
            transpose_metric_history(validation_mean_metrics),
        )
        Logger.pickle("metadata_test_raw_metrics.blob", test_raw_metrics)
        Logger.pickle(
            "metadata_test_mean_metrics.blob",
            transpose_metric_history(test_mean_metrics),
        )
        self.wandb and wandb.finish()
        Logger.echo("Cleaning process complete.")
        Logger.stop_timer()

    def apply_fn_to_tasks(
        self,
        state: nnx.Module,
        fn: callable,
        X: np.ndarray,
        Y: np.ndarray,
        M: np.ndarray,
        Tasks: list[Task],
        label: str,  # label = "train" | "validation" | "test"
        global_epoch,
    ):
        batch_metrics = self.apply_fn_to_data_with_prefetch(
            state, fn, X, Y, M, Tasks, self.sharding, global_epoch
        )
        batch_metrics = reduce_batch_metrics(batch_metrics)
        mean_metrics = mean_batch_metrics(batch_metrics)
        flatten_metrics = flatten_mean_metrics(mean_metrics, label)
        return batch_metrics, mean_metrics, flatten_metrics

    @staticmethod
    def apply_fn_to_data_with_prefetch(
        state: nnx.Module,
        fn: callable,
        X: jnp.ndarray,  # (num_steps, T, B, F)
        Y: jnp.ndarray,  # (num_steps, T, B, F)
        Mask: jnp.ndarray,  # (num_steps, T, B, F)
        Tasks: list[Task],  # (num_steps,)
        sharding,
        global_epoch,
    ):
        """
        Apply fn to data ndarray and return evaluation metric logs. Notice the order of batch_metrics is the order of batches executed where each batch will have a task associated with it.

        :param state: training state
        :type state: nnx.Module
        :param fn: function
        :type fn: callable
        :param X: data
        :type X: jnp.ndarray (num_steps, T, B, F)
        :param Y: label
        :type Y: jnp.ndarray (num_steps, T, B, F)
        :param Mask: mask
        :type Mask: jnp.ndarray (num_steps, T, B, F)
        :param Tasks: The order of tasks associated with the num_steps
        :type Tasks: list[Task]
        :param sharding: device sharding
        :param global_epoch: global epoch
        """
        batch_metrics = []

        batch_length = len(X)
        it = zip(X, Y, Mask)

        # ---- preload first batch ----
        try:
            batch_data, batch_label, batch_mask = next(it)
        except StopIteration:
            return None  # empty

        next_data = jax.device_put(batch_data, sharding)
        next_label = jax.device_put(batch_label, sharding)
        next_mask = jax.device_put(batch_mask, sharding)

        batch_count = 0
        for batch_data, batch_label, batch_mask in it:
            # swap
            cur_data, cur_label, cur_mask = next_data, next_label, next_mask

            # ---- async prefetch next batch ----
            next_data = jax.device_put(batch_data, sharding)
            next_label = jax.device_put(batch_label, sharding)
            next_mask = jax.device_put(batch_mask, sharding)

            # ---- compute on current batch ----
            batch_loss, batch_predict, batch_model_metric = fn(
                state,
                cur_data,
                cur_label,
                cur_mask,
                global_epoch,
                batch_count,
                batch_length,
            )
            task = Tasks[batch_count]
            batch_task_metric = task.evaluator(batch_predict, cur_label, cur_mask)
            batch_count += 1

            batch_metrics.append(
                {
                    "task_key": task.key,
                    "loss": batch_loss,
                    "task": batch_task_metric,
                    "model": batch_model_metric,
                }
            )

        # ---- last batch ----
        batch_loss, batch_predict, batch_model_metric = fn(
            state,
            next_data,
            next_label,
            next_mask,
            global_epoch,
            batch_count,
            batch_length,
        )
        task = Tasks[batch_count]
        batch_task_metric = task.evaluator(batch_predict, next_label, next_mask)

        batch_metrics.append(
            {
                "task_key": task.key,
                "loss": batch_loss,
                "task": batch_task_metric,
                "model": batch_model_metric,
            }
        )

        return batch_metrics
