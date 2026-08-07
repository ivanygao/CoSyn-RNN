from pathlib import Path
import os
import pickle
import psutil
import yaml
import time

RUNTIME_ROOT = "./runtime"
DATA_ROOT = os.path.join(RUNTIME_ROOT, "data")
PIPELINE_ROOT = os.path.join(RUNTIME_ROOT, "pipeline")


class Logger:
    timer = None
    prefix = None
    root = ""

    # =========================
    # Timer
    # =========================

    @classmethod
    def start_timer(cls):
        """Start or reset the timer."""
        cls.timer = time.perf_counter()

    @classmethod
    def stop_timer(cls):
        """Stop the timer, return the elapsed time, and reset it."""
        if cls.timer is None:
            return None

        elapsed = time.perf_counter() - cls.timer
        cls.timer = None
        return elapsed

    @classmethod
    def get_elapsed(cls):
        """Return the elapsed time without changing the timer state."""
        if cls.timer is None:
            return None

        return time.perf_counter() - cls.timer

    # =========================
    # Logger
    # =========================

    @classmethod
    def set_prefix(cls, prefix: str | None):
        """
        Set a prefix for stdout printing.
        If prefix is None or empty, disable prefix.
        """
        cls.prefix = prefix

    @classmethod
    def set_root(cls, root):
        cls.root = Path(root)
        cls.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def echo(cls, content):
        elapsed = cls.get_elapsed()

        # time prefix string
        time_str = f"[{elapsed:.2f}s] " if elapsed is not None else ""

        if cls.prefix:
            print(f"{time_str}[{cls.prefix}] {content}", flush=True)
        else:
            print(f"{time_str}{content}", flush=True)

    @classmethod
    def echo_and_log(cls, file_path, content):
        """
        store content as utf-8 text
        """
        file_path = Path(cls.root / file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        cls.echo(content)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    @classmethod
    def log_yaml(cls, file_path, content):
        """
        store content as utf-8 text
        """
        file_path = Path(cls.root / file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(content.model_dump(mode="json"), f, sort_keys=False)

    @classmethod
    def pickle(cls, file_path, content):
        """
        store content as BLOB
        """
        file_path = Path(cls.root / file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(
            file_path,
            "wb",
        ) as f:
            pickle.dump(content, f)

    @classmethod
    def get_total_ram_usage(cls):
        # total memory: prefer SLURM_MEM_PER_NODE if available
        if "SLURM_MEM_PER_NODE" in os.environ and os.environ["SLURM_MEM_PER_NODE"]:
            # SLURM_MEM_PER_NODE is in MB
            total_gb = int(os.environ["SLURM_MEM_PER_NODE"]) / 1024
        else:
            total_gb = psutil.virtual_memory().total / (1024**3)

        return total_gb

    @classmethod
    def get_ram_usage(cls):
        """
        Return RAM usage as a compact string:
        'X.XX / Y.YY GB (Z.ZZ%)'
        """
        p = psutil.Process(os.getpid())
        rss_gb = p.memory_info().rss / (1024**3)

        total_gb = round(cls.get_total_ram_usage(), 2)
        percent = 100.0 * rss_gb / total_gb if total_gb > 0 else 0.0

        return f"{rss_gb:.2f}/{total_gb:.2f} GB ({percent:.2f}%)"
