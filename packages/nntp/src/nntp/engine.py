import os, psutil, platform, GPUtil

import sys
import subprocess
import threading
import queue
import time
from pathlib import Path

from .logger import Logger
from .experiment import Experiment


class Engine:
    def __str__(self) -> str:
        lines = []
        lines.append(f"{'='*40}")
        lines.append(f"\tEngine Summary ")
        lines.append(f"Platform Name  : {self.name or 'Unknown'}")
        lines.append(f"Architecture   : {self.architecture or 'Unknown'}")
        lines.append(f"CPU Cores      : {self.CPU_cores or 'Unknown'}")
        lines.append(
            f"Memory (GB)    : {self.memory_GB if self.memory_GB is not None else 'Unknown'}"
        )

        if self.GPUs:
            lines.append("GPUs Detected  :")
            for gpu in self.GPUs:
                lines.append(f"  - {gpu}")
        else:
            lines.append("GPUs Detected  : None")

        lines.append("=" * 40)
        return "\n".join(lines)

    def __init__(self, entrypoint: str | Path, group):
        self.group = group
        self.entrypoint = Path(entrypoint).expanduser().resolve()

        # detecting platform metadata
        self.name = platform.system()
        self.architecture = platform.machine()
        self.CPU_cores = os.cpu_count() or psutil.cpu_count(logical=True)
        self.memory_GB = Logger.get_total_ram_usage()

        gpus = GPUtil.getGPUs()
        if not gpus:
            raise RuntimeError(
                "No GPU detected. Engine requires at least one visible GPU."
            )

        self.GPUs = [
            (
                f"GPU: {gpu.name}, "
                f"Memory {gpu.memoryTotal} MB, "
                f"Load {gpu.load * 100:.1f}%"
            )
            for gpu in gpus
        ]

        Logger.set_prefix("Engine")
        Logger.echo_and_log("metadata_engine.txt", str(self))

        # enable async thread to lock on gpu so one task per gpu
        self.experiment_queue = queue.Queue()
        self.pending_experiment = 0
        self.pending_lock = threading.Lock()
        self.worker_threads = []
        self.error_event = threading.Event()

        # start worker threads (1 thread per GPU)
        for gpu_id in range(len(self.GPUs)):
            t = threading.Thread(
                target=self._worker,
                args=(gpu_id,),
                daemon=True,
            )
            t.start()
            self.worker_threads.append(t)

        Logger.echo(f"Started {len(self.worker_threads)} worker threads.")

    def _worker(self, worker_id: int):
        """Each worker thread is bound to exactly one GPU (worker_id)."""

        while True:
            item = self.experiment_queue.get()
            if item is None:
                self.experiment_queue.task_done()
                break

            experiment, group = item
            start = time.time()
            stdout_f = None
            stderr_f = None

            try:
                experiment_dir = Logger.root / str(experiment.seed) / experiment.name
                stdout_path = experiment_dir / "stdout.log"
                stderr_path = experiment_dir / "stderr.log"

                Logger.echo(
                    f"Worker {worker_id} started experiment "
                    f"{experiment.name} "
                    f"[{self.pending_experiment} pending]. "
                    f"Output redirected to {stdout_path}"
                )

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
                env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

                cmd = [
                    sys.executable,
                    "-m",
                    "nntp.bootstrap",
                    "--entrypoint",
                    str(self.entrypoint),
                    "worker",
                    "--id",
                    str(worker_id),
                    "--name",
                    experiment.name,
                    "--group",
                    group,
                    "--config",
                    str(experiment_dir / "metadata_config.yaml"),
                ]

                if experiment.debug:
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        text=True,
                    )
                else:
                    stdout_f = open(stdout_path, "a", encoding="utf-8")
                    stderr_f = open(stderr_path, "a", encoding="utf-8")
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        text=True,
                    )

                return_code = process.wait()

                if return_code != 0:
                    Logger.echo(
                        f"Experiment {experiment.name} "
                        f"failed with code {return_code}"
                    )
                    self.error_event.set()
                else:
                    Logger.echo(f"Experiment {experiment.name} completed successfully")

            except Exception as exc:
                self.error_event.set()
                Logger.echo(
                    f"Worker {worker_id} failed: " f"{type(exc).__name__}: {exc}"
                )

            finally:
                if stdout_f is not None:
                    stdout_f.close()
                if stderr_f is not None:
                    stderr_f.close()

                elapsed = time.time() - start

                with self.pending_lock:
                    self.pending_experiment -= 1
                    pending = self.pending_experiment

                Logger.echo(f"Experiment {experiment.name} done " f"({elapsed:.3f}s)")

                self.experiment_queue.task_done()

    def run(self, experiments: list[Experiment]):
        with self.pending_lock:
            self.pending_experiment += len(experiments)

        for experiment in experiments:
            self.experiment_queue.put((experiment, self.group))

    def wait(self):
        self.experiment_queue.join()

        for _ in self.worker_threads:
            self.experiment_queue.put(None)

        for t in self.worker_threads:
            t.join()

        sys.exit(1 if self.error_event.is_set() else 0)
