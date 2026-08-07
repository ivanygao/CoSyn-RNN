from jax import numpy as jnp
import numpy as np


def masked_mean(x, mask, axis=None, keepdims=True):
    sum_ = jnp.sum(x * mask, axis=axis, keepdims=keepdims)
    count = jnp.sum(mask, axis=axis, keepdims=keepdims)
    return sum_ / jnp.maximum(count, 1.0)


def compute_W_rec_task_block_cross_norms(
    W_rec_value,
    identity_value,
    norm_type="fro",
):
    """
    Compute block-wise norms of W_rec grouped by identity.

    Assumption:
        W_rec_value[pre, post]
        identity_value[H] stores ordered phase/task ids, e.g. 0, 1, 2, ...

    Args:
        W_rec_value:
            Array [H, H].
        identity_value:
            Array [H], ordered identity id for each neuron.
        norm_type:
            "fro"   -> Frobenius norm
            "l1"    -> sum(abs(W))
            "mean"  -> mean(abs(W))
            "count" -> number of nonzero entries

    Returns:
        norm_matrix:
            Array [T, T], where norm_matrix[pre, post]
            is the norm of pre -> post.
    """

    W = np.asarray(W_rec_value)
    identity = np.asarray(identity_value).astype(int)

    assert W.ndim == 2, f"Expected W_rec to be 2D, got shape {W.shape}"
    assert identity.ndim == 1, f"Expected identity to be 1D, got shape {identity.shape}"
    assert W.shape[0] == identity.shape[0], (
        f"W_rec first axis and identity length mismatch: " f"W_rec.shape={W.shape}, identity.shape={identity.shape}"
    )
    assert W.shape[1] == identity.shape[0], (
        f"W_rec second axis and identity length mismatch: " f"W_rec.shape={W.shape}, identity.shape={identity.shape}"
    )

    # identity is assumed to store ordered ids: 0, 1, 2, ...
    task_ids = np.unique(identity)
    task_ids = task_ids[task_ids >= 0]

    T = len(task_ids)
    norm_matrix = np.zeros((T, T), dtype=float)

    for pre_pos, pre_id in enumerate(task_ids):
        pre_idx = np.where(identity == pre_id)[0]

        for post_pos, post_id in enumerate(task_ids):
            post_idx = np.where(identity == post_id)[0]

            # W_rec[pre, post]
            # block: pre_id -> post_id
            block = W[np.ix_(pre_idx, post_idx)]

            if block.size == 0:
                norm_matrix[pre_pos, post_pos] = np.nan
                continue

            if norm_type == "fro":
                value_norm = np.linalg.norm(block) / block.size
            elif norm_type == "l1":
                value_norm = np.sum(np.abs(block)) / block.size
            elif norm_type == "mean":
                value_norm = np.mean(np.abs(block))
            elif norm_type == "count":
                value_norm = np.count_nonzero(block) / block.size
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")

            # norm_matrix[pre, post] = pre -> post
            norm_matrix[pre_pos, post_pos] = value_norm

    return norm_matrix
