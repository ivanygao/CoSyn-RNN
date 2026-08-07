import matplotlib.pyplot as plt
import wandb
import time
import numpy as np
import math
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap, BoundaryNorm

from ..logger import Logger


def plot_masks(C, label_names):
    # use the first n color in tab10
    keys = sorted(label_names.keys())
    n = len(keys)
    base_cmap = plt.get_cmap("tab10")
    colors = base_cmap.colors[:n]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(C, cmap=cmap, vmin=0, vmax=n - 1)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=keys)
    cbar.ax.set_yticklabels([label_names[k] for k in keys])

    ax.set_xlabel("neurons")
    ax.set_ylabel("neurons")
    ax.set_title("W_rec Masks")

    fig.tight_layout()
    return fig


def plot_eigenvalue_spectrum(W_rec):
    eigvals = np.linalg.eigvals(W_rec)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # eigenvalues
    ax.scatter(eigvals.real, eigvals.imag, s=10, alpha=0.7)

    # unit circle
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "r--", linewidth=1)

    # axes
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("Eigenvalue Spectrum of W_rec")

    fig.tight_layout()
    return fig


def plot_matrices_heatmap_helper(W, path, title):
    start_time = time.time()
    Logger.echo(f"Plotting {title} heatmap...")
    # ==============================================
    if W.ndim == 1:
        W = W[:, None]  # (n,) -> (n,1)
    elif W.ndim == 0:
        W = W.reshape(1, 1)  # scaler -> (1,1)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(W, aspect="auto", cmap="viridis")
    ax.set_xlabel("Presynaptic neurons")
    ax.set_ylabel("Postsynaptic neuron")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title("Scaled value")
    fig.tight_layout()

    fig.savefig(path / f"{title}.png")
    matrices_heatmap = wandb.Image(fig)
    plt.close(fig)
    # ==============================================
    Logger.echo(
        f"{title} heatmap plotted. Time taken: {time.time() - start_time:.2f} seconds."
    )
    return matrices_heatmap


def plot_matrices_histogram_helper(Ws: dict, path, title):
    start_time = time.time()
    Logger.echo(f"Plotting {title} histogram...")
    # ==============================================
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    for name, W in Ws.items():
        arr = np.asarray(W, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            continue

        ax.hist(arr, bins=W.shape[0], alpha=0.5, label=name)

    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(f"{title} (Log Count)")
    ax.legend()
    fig.tight_layout()

    fig.savefig(path / f"{title}.png")
    matrices_histogram = wandb.Image(fig)
    plt.close(fig)
    # ==============================================
    Logger.echo(
        f"{title} histogram plotted. Time taken: {time.time() - start_time:.2f} seconds."
    )
    return matrices_histogram


def plot_task_block_norm_heatmap_helper(norm_matrix, title, path):
    """
    Plot task-block norm heatmap.

    norm_matrix[pre, post] means:
        pre -> post.

    Therefore:
        x-axis = pre/source id
        y-axis = post/target id.
    """

    start_time = time.time()
    Logger.echo(f"Plotting {title} task block norm heatmap...")
    # ==============================================

    norm_matrix = np.asarray(norm_matrix)

    assert (
        norm_matrix.ndim == 2
    ), f"Expected 2D norm_matrix, got shape {norm_matrix.shape}"
    assert (
        norm_matrix.shape[0] == norm_matrix.shape[1]
    ), f"Expected square norm_matrix, got shape {norm_matrix.shape}"

    T = norm_matrix.shape[0]

    if T == 0:
        Logger.echo(
            f"Skipping {title}: empty norm_matrix with shape {norm_matrix.shape}"
        )
        return None

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    im = ax.imshow(norm_matrix, aspect="auto", cmap="viridis_r")

    ax.set_xticks(np.arange(T))
    ax.set_yticks(np.arange(T))

    ax.set_xticklabels(
        [f"id {i}" for i in range(T)], rotation=45, ha="right", fontsize=8
    )
    ax.set_yticklabels([f"id {i}" for i in range(T)], fontsize=8)

    ax.set_xlabel("Pre/source task id", fontsize=9)
    ax.set_ylabel("Post/target task id", fontsize=9)
    ax.set_title(title, fontsize=10)

    ax.tick_params(axis="both", labelsize=8)

    # Annotate values.
    # norm_matrix[post_id, pre_id] = pre_id -> post_id
    for post_id in range(T):
        for pre_id in range(T):
            value = norm_matrix[post_id, pre_id]
            if np.isfinite(value):
                ax.text(
                    pre_id,  # x = pre/source
                    post_id,  # y = post/target
                    f"{value:.2e}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.colorbar(im, ax=ax, label="norm")
    fig.tight_layout()
    fig.savefig(path / f"{title}.png")

    task_block_norm_heatmap = wandb.Image(fig)
    plt.close(fig)

    # ==============================================
    Logger.echo(
        f"{title} Task block norm heatmap plotted. Time taken: {time.time() - start_time:.2f} seconds."
    )
    return task_block_norm_heatmap


def plot_task_block_sorted_by_threshold_count_heatmap_helper(
    W_value, identity_value, threshold, path, title
):
    start_time = time.time()
    Logger.echo(f"Plotting {title} task block sorted by threshold count heatmap...")
    # ==============================================

    W_value = np.asarray(W_value)
    identity_value = np.asarray(identity_value).astype(int)

    assert W_value.ndim == 2, f"Expected W_value to be 2D, got shape {W_value.shape}"
    assert (
        identity_value.ndim == 1
    ), f"Expected identity_value to be 1D, got shape {identity_value.shape}"
    assert W_value.shape[0] == identity_value.shape[0], (
        f"W_value first axis and identity_value length mismatch: "
        f"W_value.shape={W_value.shape}, identity_value.shape={identity_value.shape}"
    )
    assert W_value.shape[1] == identity_value.shape[0], (
        f"W_value second axis and identity_value length mismatch: "
        f"W_value.shape={W_value.shape}, identity_value.shape={identity_value.shape}"
    )

    active_mask = np.abs(W_value) > threshold

    # identity is assumed to store ordered ids: 0, 1, 2, ...
    task_ids = np.unique(identity_value)
    task_ids = task_ids[task_ids >= 0]

    orders = []
    block_labels = []

    for task_id in task_ids:
        task_indices = np.where(identity_value == task_id)[0]

        if len(task_indices) == 0:
            continue

        # Only look at within-task active connections.
        block_active = active_mask[np.ix_(task_indices, task_indices)]

        block_row_count = active_mask[task_indices, :].sum(axis=1)
        block_col_count = block_active.sum(axis=0)
        block_score = block_row_count + block_col_count

        task_order = task_indices[np.argsort(-block_score)]
        orders.append(task_order)
        block_labels.append(task_id)

    if len(orders) == 0:
        Logger.echo(f"Skipping {title}: no valid task ids found in identity_value.")
        return None

    order = np.concatenate(orders)

    # ==============================================
    # plot
    # ==============================================
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    W_sorted = W_value[np.ix_(order, order)]

    # ------------------------------------------------
    # Continuous signed heatmap.
    # Show clipped real values in [-threshold, threshold].
    # ------------------------------------------------
    W_display = np.clip(W_sorted, -threshold, threshold)

    if np.all(np.isnan(W_display)):
        Logger.echo(f"Skipping {title}: all values are NaN.")
        plt.close(fig)
        return None

    vmin = np.nanmin(W_display)
    vmax = np.nanmax(W_display)

    im = ax.imshow(
        W_display,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=vmin,
            vcenter=0.0,
            vmax=vmax,
        ),
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.08)
    cbar.set_label("clipped weight", fontsize=8)

    # ==============================================
    # Block boundaries
    # ==============================================
    block_sizes = [len(x) for x in orders]
    boundaries = np.cumsum(block_sizes)[:-1]

    for b in boundaries:
        ax.axhline(b - 0.5, color="black", linewidth=0.8)
        ax.axvline(b - 0.5, color="black", linewidth=0.8)

    # ------------------------------------------------
    # Main ticks: keep sorted neuron positions 0..H
    # ------------------------------------------------
    H_sorted = len(order)

    num_main_ticks = min(9, H_sorted)
    main_ticks = np.linspace(0, H_sorted - 1, num_main_ticks, dtype=int)

    ax.set_xticks(main_ticks)
    ax.set_yticks(main_ticks)

    ax.set_xticklabels([str(t) for t in main_ticks], fontsize=8)
    ax.set_yticklabels([str(t) for t in main_ticks], fontsize=8)

    # ------------------------------------------------
    # Secondary axes: show block identity labels
    # ------------------------------------------------
    block_starts = np.concatenate([[0], np.cumsum(block_sizes)[:-1]])
    block_centers = block_starts + np.asarray(block_sizes) / 2 - 0.5
    block_ticklabels = [f"id {task_id}" for task_id in block_labels]

    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(block_centers)
    ax_top.set_xticklabels(block_ticklabels, rotation=45, ha="left", fontsize=8)

    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks(block_centers)
    ax_right.set_yticklabels(block_ticklabels, fontsize=8)

    # ------------------------------------------------
    # Axis labels
    # ------------------------------------------------
    ax.set_xlabel("Presynaptic neurons (Sorted)")
    ax.set_ylabel("Postsynaptic neurons (Sorted)")
    ax.set_title(title)

    ax.tick_params(axis="both")

    fig.tight_layout()

    fig.savefig(path / f"{title}.png")

    rec_sorted_by_threshold_count_heatmap = wandb.Image(fig)
    plt.close(fig)

    # ==============================================
    Logger.echo(
        f"{title} Task block sorted by threshold count heatmap plotted. "
        f"Time taken: {time.time() - start_time:.2f} seconds."
    )
    return rec_sorted_by_threshold_count_heatmap


def plot_W_grid_helper(W, channel, threshold, path, title):
    """
    Plot all slices of a 3D tensor as a grid of 2D heatmaps by slicing along axis 0.

    Parameters
    ----------
    W : array-like
        3D array, e.g. [C, H, O].
    path : pathlib.Path
        Save directory.
    title : str
        Figure/file title.
    """

    start_time = time.time()
    Logger.echo(f"Plotting {title} W grid...")
    # ==============================================

    W = np.asarray(W)

    selected_ids = np.where(channel)[0]
    W = W[channel]

    if W.shape[0] == 0:
        Logger.echo(f"Skipping {title}: no active channels.")
        return None

    if np.all(np.isnan(W)):
        Logger.echo(f"Skipping {title}: all values are NaN.")
        return None

    num_slices, _, _ = W.shape

    ncols = math.ceil(math.sqrt(num_slices))
    nrows = math.ceil(num_slices / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
        constrained_layout=True,
        dpi=300,
    )

    for slice_id in range(num_slices):
        r = slice_id // ncols
        c = slice_id % ncols
        ax = axes[r][c]

        img = W[slice_id].T

        active_mask = np.abs(img) > threshold
        # Sort columns by absolute column sum, high to low.
        col_score = np.sum(active_mask, axis=0)
        col_order = np.argsort(-col_score)

        img = img[:, col_order]

        img = np.clip(img, -threshold, threshold)

        vmin = np.nanmin(img)
        vmax = np.nanmax(img)

        norm = TwoSlopeNorm(
            vmin=vmin,
            vcenter=0.0,
            vmax=vmax,
        )

        im = ax.imshow(
            img,
            aspect="auto",
            cmap="bwr_r",
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(f"id {selected_ids[slice_id]}", fontsize=8)

        ax.set_xlabel("Neurons (sorted)", fontsize=7)
        ax.set_ylabel("Output", fontsize=7)

        ax.tick_params(axis="both", labelsize=6)
        fig.colorbar(im, ax=ax, pad=0.02)

    for idx in range(num_slices, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=10)

    fig.savefig(path / f"{title}.png")
    W_grid = wandb.Image(fig)
    plt.close(fig)

    # ==============================================
    Logger.echo(
        f"{title} W grid plotted. Time taken: {time.time() - start_time:.2f} seconds."
    )
    return W_grid
