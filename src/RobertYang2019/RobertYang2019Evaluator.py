from jax import vmap
from jax import numpy as jnp
from flax import nnx

from .RobertYang2019Decoder import ring_decoder


@nnx.jit
def circular_distance(predicts, targets):
    """
    Compute minimal absolute angular distance between predicts and targets
    on a circular domain [-pi, pi).
    """
    return jnp.abs((predicts - targets + jnp.pi) % (2 * jnp.pi) - jnp.pi)


@nnx.jit
def apply_abs_diff_with_mask(tol, predicts, targets, masks, valid):
    """
    input: (T, B)
    """
    numerator = jnp.sum((jnp.abs(predicts - targets) < tol) * masks, axis=0)
    denominator = jnp.sum(masks, axis=0)
    numerator = jnp.where(valid, numerator, 0.0)
    denominator = jnp.where(valid, denominator, 1.0)
    return jnp.sum(numerator / denominator) / jnp.maximum(jnp.sum(valid), 1)


@nnx.jit
def apply_circular_diff_with_mask(tol, predicts, targets, masks, valid):
    """
    input: (T, B)
    """
    numerator = jnp.sum((circular_distance(predicts, targets) < tol) * masks, axis=0)
    denominator = jnp.sum(masks, axis=0)
    numerator = jnp.where(valid, numerator, 0.0)
    denominator = jnp.where(valid, denominator, 1.0)
    return jnp.sum(numerator / denominator) / jnp.maximum(jnp.sum(valid), 1)


@nnx.jit
def apply_abs_diff_without_mask(tol, predicts, targets, valid):
    """
    input: (B,)
    """
    numerator = jnp.where(valid, (jnp.abs(predicts - targets) < tol), 0.0)
    return jnp.sum(numerator) / jnp.maximum(jnp.sum(valid), 1)


@nnx.jit
def apply_circular_diff_without_mask(tol, predicts, targets, valid):
    """
    input: (B,)
    """
    numerator = jnp.where(valid, circular_distance(predicts, targets) < tol, 0.0)
    return jnp.sum(numerator) / jnp.maximum(jnp.sum(valid), 1)


def build_evaluator(cfg):
    # fixation_tolerance = cfg["fixation_tolerance"]
    # radius_tolerance = cfg["radius_tolerance"]
    # angle_tolerance = cfg["angle_tolerance"]
    # angle_tolerance = angle_tolerance * jnp.pi / 180  # result in radian
    fixation_threshold = cfg["fixation_threshold"]
    radius_threshold = cfg["radius_threshold"]
    angle_threshold = cfg["angle_threshold"]
    angle_threshold = angle_threshold * jnp.pi / 180  # result in radian

    @nnx.jit
    def evaluator(predict, Y, M):
        # identify padding trails (if mask is on then trail is valid)
        T, B, _ = M.shape
        valid_trails = jnp.sum(M, axis=(0, 2)) > 0  # (B,)

        # compute consistency
        fixation_predict = predict[:, :, 0]  # (T, B)
        fixation_label = Y[:, :, 0]  # (T, B)
        # fixation_mask_decode = M[:, :, 0] > 0  # (T, B)
        ring_label = Y[:, :, 1:]  # (T, B, O)
        ring_predict = predict[:, :, 1:]  # (T, B, O)
        # ring_mask_decode = jnp.sum(M[:, :, 1:], axis=2) > 0  # (T, B)

        # compute fixation consistency
        # consistency_fixation = apply_abs_diff_with_mask(
        #     fixation_tolerance, fixation_predict, fixation_label, fixation_mask_decode, valid_trails
        # )

        # compute radius and angle consistency
        radius_predict_decode, angle_predict_decode = vmap(ring_decoder, in_axes=1, out_axes=1)(ring_predict)
        radius_label_decode, angle_label_decode = vmap(ring_decoder, in_axes=1, out_axes=1)(ring_label)
        # consistency_radius = apply_abs_diff_with_mask(
        #     radius_tolerance, radius_predict_decode, radius_label_decode, ring_mask_decode, valid_trails
        # )
        # consistency_angle = apply_circular_diff_with_mask(
        #     angle_tolerance, angle_predict_decode, angle_label_decode, ring_mask_decode, valid_trails
        # )

        # compute accuracy

        # locate the last output of each trail
        mask_bool = M[:, :, 0] > 0  # [T, B]
        rev = mask_bool[::-1, :]  # reverse time T, [T, B]
        idx_from_end = jnp.argmax(rev, axis=0)  # (B,)
        last_index = T - 1 - idx_from_end  # (B,)

        # compute fixation accuracy and extract the last output from each trail
        accuracy_fixation = apply_abs_diff_without_mask(
            fixation_threshold,
            fixation_predict[last_index, jnp.arange(B)],
            fixation_label[last_index, jnp.arange(B)],
            valid_trails,
        )

        # compute radius accuracy
        accuracy_radius = apply_abs_diff_without_mask(
            radius_threshold,
            radius_predict_decode[last_index, jnp.arange(B)],
            radius_label_decode[last_index, jnp.arange(B)],
            valid_trails,
        )

        # compute angle accuracy
        accuracy_angle = apply_circular_diff_without_mask(
            angle_threshold,
            angle_predict_decode[last_index, jnp.arange(B)],
            angle_label_decode[last_index, jnp.arange(B)],
            valid_trails,
        )

        return {
            # "consistency_fixation": consistency_fixation,
            # "consistency_radius": consistency_radius,
            # "consistency_angle": consistency_angle,
            "accuracy_fixation": accuracy_fixation,
            "accuracy_radius": accuracy_radius,
            "accuracy_angle": accuracy_angle,
        }

    return evaluator
