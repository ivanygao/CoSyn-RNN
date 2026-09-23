import jax
from jax import numpy as jnp
from jax import random as jrand
from jax import lax, jit
from flax import nnx
from functools import partial

from nntp.models.enums_getter import get_activation_function
from nntp.models.enums import Distribution


@partial(jit, static_argnames=["hidden_activation", "output_activation"])
def forward(
    key,
    X,
    W_in,
    W_rec,
    W_out,
    b_h,
    b_y,
    dt,
    tau,
    hidden_activation,
    output_activation,
    noise_std,
):
    """
    X: [T, B, F]  (trail_length, batch_size, feature_size)
    return: [T, B, O] (trail_length, batch_size, output_size)
    """
    alpha = dt / tau
    scale = jnp.sqrt((2 - alpha) / alpha)
    noise_scale = noise_std * scale

    T, B, _ = X.shape
    H = b_h.shape[0]

    def step(old_h, inputs):  # X_t: [B, F], noise_t: [B, H]
        X_t, noise_t = inputs
        new_h = (1 - alpha) * old_h + alpha * hidden_activation(
            X_t @ W_in + old_h @ W_rec + b_h + noise_t
        )  # [B, H]
        return new_h, new_h

    noise = jax.random.normal(key, (T, B, H)) * noise_scale
    _, h = lax.scan(step, jnp.zeros((B, H)), (X, noise))  # h: [T, B, H]
    return output_activation(h @ W_out + b_y)  # y: [T, B, output_size]


class RNN(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        feature_size,
        output_size,
        cfg,
    ):
        self.rngs = rngs
        self.feature_size = feature_size
        self.output_size = output_size
        self.dt = cfg["dt"]
        self.tau = cfg["tau"]
        self.hidden_size = cfg["hidden_size"]
        self.hidden_activation = get_activation_function(cfg["hidden_activation"])
        self.output_activation = get_activation_function(cfg["output_activation"])
        self.noise_std = cfg.get("noise_std", None)

        self.initial_weight_distribution = cfg["initial_weight_distribution"]
        self.initial_spectral_radius = cfg.get("initial_spectral_radius", None)

        # initialize weights
        if cfg.get("trainable_params", {}).get("W_in", False):
            self.W_in = nnx.Param(
                self.initial_weight((self.feature_size, self.hidden_size))
            )
        else:
            self.W_in = nnx.Variable(
                self.initial_weight((self.feature_size, self.hidden_size))
            )

        if cfg.get("trainable_params", {}).get("W_rec", False):
            self.W_rec = nnx.Param(
                self.initial_weight(
                    (self.hidden_size, self.hidden_size), self.initial_spectral_radius
                )
            )
        else:
            self.W_rec = nnx.Variable(
                self.initial_weight(
                    (self.hidden_size, self.hidden_size), self.initial_spectral_radius
                )
            )

        if cfg.get("trainable_params", {}).get("W_out", False):
            self.W_out = nnx.Param(
                self.initial_weight((self.hidden_size, self.output_size))
            )
        else:
            self.W_out = nnx.Variable(
                self.initial_weight((self.hidden_size, self.output_size))
            )

        if cfg.get("trainable_params", {}).get("b_h", False):
            self.b_h = nnx.Param(self.initial_weight((self.hidden_size,)))
        else:
            self.b_h = nnx.Variable(self.initial_weight((self.hidden_size,)))

        if cfg.get("trainable_params", {}).get("b_y", False):
            self.b_y = nnx.Param(self.initial_weight((self.output_size,)))
        else:
            self.b_y = nnx.Variable(self.initial_weight((self.output_size,)))

    def initial_weight(self, shape, target_spectral_radius=None):
        """
        by random matrix theory:
        weight ~ Normal (0, sigma = 1/sqrt(N)) will give a matrix with max(abs(eigen-value)) ~ 1
        weight ~ Uniform (-sqrt(3/N), sqrt(3/N)) will give a matrix with max(abs(eigen-value)) ~ 1
        """
        in_size = shape[0]
        std = 1 / jnp.sqrt(in_size)

        if self.initial_weight_distribution == Distribution.NORMAL:
            W = std * jrand.normal(self.rngs(), shape)
        elif self.initial_weight_distribution == Distribution.UNIFORM:
            # Scale for uniform distribution to match normal std which is 1/sqrt(feature_size)
            a = jnp.sqrt(3) * std
            W = jrand.uniform(self.rngs(), shape, minval=-a, maxval=+a)

        if target_spectral_radius is not None and shape[0] == shape[1]:
            # compute spectral norm
            eigvals = jnp.linalg.eigvals(W)
            W = W * (target_spectral_radius / jnp.max(jnp.abs(eigvals)))
        return W

    def __call__(self, X):
        return forward(
            self.rngs(),
            X,
            self.W_in,
            self.W_rec,
            self.W_out,
            self.b_h,
            self.b_y,
            self.dt,
            self.tau,
            self.hidden_activation,
            self.output_activation,
            self.noise_std,
        )

    def evaluation(self, X):
        return forward(
            self.rngs(),
            X,
            self.W_in,
            self.W_rec,
            self.W_out,
            self.b_h,
            self.b_y,
            self.dt,
            self.tau,
            self.hidden_activation,
            self.output_activation,
            self.noise_std,
        )
