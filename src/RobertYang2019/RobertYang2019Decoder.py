from jax import numpy as jnp
from flax import nnx


@nnx.jit
def ring_decoder(ring_vector):
    """
    Decode radius and angle from population encoding.

    :param ring_vector: [T, F] where F=32 is number of ring neurons
    :return: angle in radians, shape [T], range [0, 2π)
    """
    preference = jnp.arange(0, 2 * jnp.pi, 2 * jnp.pi / 32)
    radius = jnp.max(ring_vector, axis=1)
    angle = jnp.arctan2(
        jnp.sum(ring_vector * jnp.sin(preference), axis=1),
        jnp.sum(ring_vector * jnp.cos(preference), axis=1),
    ) % (2 * jnp.pi)
    return radius, angle
