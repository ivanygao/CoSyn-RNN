import jax
from jax import numpy as jnp
from jax import random as jrand
from jax import lax, jit
from flax import nnx
from functools import partial

from nntp.models.enums_getter import get_activation_function
from nntp.models.enums import Distribution


@partial(
    jit,
    static_argnames=["hidden_activation", "modulation_activation", "output_activation"],
)
def forward(
    key,
    X_cue,  # [T, B, C]
    X,  # [T, B, F]
    W_in,  # [F, H]
    W_rec,  # [H, H]
    W_out,  # [H, O]
    b_h,  # [H]
    b_y,  # [O]
    excitation,  # [H]
    W_mask,  # [C, H, O]
    dt,
    tau,
    hidden_activation,
    modulation_activation,
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

    excitation = jax.nn.relu(excitation)

    def step(old_h, inputs):  # X_t: [B, F], noise_t: [B, H]
        X_t, noise_t = inputs
        new_h = (1 - alpha) * old_h + alpha * hidden_activation(
            X_t @ W_in + old_h @ W_rec + b_h[None, :] + noise_t
        )  # [B, H]
        new_h = new_h * excitation[None, :]
        return new_h, new_h

    noise = jax.random.normal(key, (T, B, H)) * noise_scale
    _, h = lax.scan(
        step, jnp.zeros((B, H)), (X, noise)
    )  # X: [T, B, F], noise: [T, B, H] => h: [T, B, H], in_h: [T, B, H]

    # X_cue: [T, B, C], W_mask: [C, H, O] => raw modulation of output per T, B: [T, B, H，O]
    modulation = modulation_activation(jnp.einsum("tbc,cho->tbho", X_cue, W_mask))

    # W_out [H, O], modulation [T, B, H, O] => W_out_modulated: [T, B, H, O]
    W_out_modulated = W_out[None, None, :, :] * modulation

    # h [T, B, H], W_out_modulated [T, B, H, O] => readout: [T, B, O]
    readout = jnp.einsum("tbh,tbho->tbo", h, W_out_modulated)

    y = output_activation(readout + b_y)  # y: [T, B, O]
    return y, W_out_modulated


class CoSynRNN(nnx.Module):
    # ==============================
    # All naming of row and col are respect to theoritical formula which is the transpose of its actual implementation.
    # ==============================
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
        self.excitation_activation = get_activation_function(
            cfg["excitation_activation"]
        )
        self.modulation_activation = get_activation_function(
            cfg["modulation_activation"]
        )
        self.output_activation = get_activation_function(cfg["output_activation"])
        self.initial_weight_distribution = cfg["initial_weight_distribution"]
        self.initial_spectral_radius = cfg["initial_spectral_radius"]
        self.noise_std = cfg.get("noise_std", None)

        self.cue_index_start = cfg["cue_index_start"]
        self.threshold = cfg["threshold"]

        # derived variables
        self.C = self.feature_size - self.cue_index_start
        self.scale = self.get_scale(self.hidden_size)

        # parameters
        self.W_in = nnx.Param(
            self.initial_weight((self.feature_size, self.hidden_size))
        )
        self.W_rec = nnx.Param(
            self.initial_weight(
                (self.hidden_size, self.hidden_size), self.initial_spectral_radius
            )
        )
        self.W_out = nnx.Variable(
            self.initial_weight((self.hidden_size, self.output_size))
        )
        self.b_h = nnx.Param(self.initial_weight((self.hidden_size,)))
        self.b_y = nnx.Variable(self.initial_weight((self.output_size,)))
        self.W_mask = nnx.Param(jnp.zeros((self.C, self.hidden_size, self.output_size)))

        self.cue = nnx.Variable(
            jnp.array(-1, dtype=jnp.int32)
        )  # pass along cue for consolidation
        self.trainable_mask = nnx.Variable(
            jnp.ones((self.hidden_size,), dtype=jnp.bool_)
        )  # indicating which neurons are trainable
        self.excitation = nnx.Param(jnp.ones((self.hidden_size,)))

        # plotting variable
        self.channel = nnx.Variable(
            jnp.zeros((self.C,), dtype=jnp.bool_)
        )  # This indicates which cue has shown up. For plotting and sconsolidation
        self.identity = nnx.Variable(
            jnp.zeros((self.hidden_size,), dtype=jnp.int32)
        )  # identity = 0, 1, 2, etc corrsponding to task order. This indicates which neuron corresponds to which cue. For plotting

    def get_scale(self, size):
        if self.initial_weight_distribution == Distribution.NORMAL:
            return jnp.sqrt(1 / size)
        elif self.initial_weight_distribution == Distribution.UNIFORM:
            return jnp.sqrt(3 / size)

    def initial_weight(self, shape, target_spectral_radius=None):
        """
        by random matrix theory:
        weight ~ Normal (0, sigma = 1/sqrt(N)) will give a matrix with max(abs(eigen-value)) ~ 1
        weight ~ Uniform (-sqrt(3/N), sqrt(3/N)) will give a matrix with max(abs(eigen-value)) ~ 1
        """
        scale = self.get_scale(shape[0])

        if self.initial_weight_distribution == Distribution.NORMAL:
            W = jrand.normal(self.rngs(), shape) * scale
        elif self.initial_weight_distribution == Distribution.UNIFORM:
            # Scale for uniform distribution to match normal std which is 1/sqrt(feature_size)
            W = jrand.uniform(self.rngs(), shape, minval=-scale, maxval=+scale)

        if target_spectral_radius is not None and shape[0] == shape[1]:
            # compute spectral radius
            eigvals = jnp.linalg.eigvals(W)
            W = W * (target_spectral_radius / jnp.max(jnp.abs(eigvals)))
        return W

    def get_cue_index(self, X):
        return jnp.argmax(X[0, 0, self.cue_index_start :], axis=-1)

    def get_X_cue_TBC(self, X):
        return jax.nn.one_hot(
            jnp.argmax(X[:, :, self.cue_index_start :], axis=-1), self.C
        )

    def __call__(self, X, batch_count, epoch):
        W_rec_value = self.W_rec.get_value()
        W_out_value = self.W_out.get_value()
        W_mask_value = self.W_mask.get_value()
        cue_value = self.cue.get_value()
        threshold = self.threshold

        # ==============================
        # 1. Consolidation
        # ==============================
        channel_value = self.channel.get_value()
        cue_index = self.get_cue_index(X)

        is_new_cue = (batch_count == 0) & (~channel_value[cue_index])
        should_consolidate = is_new_cue & (epoch > 1)

        def consolidate(trainable_mask, excitation_value, identity):
            # recurrent
            W_rec_active_synapse = jnp.abs(W_rec_value) > threshold
            recurrent_free_neurons = (
                (W_rec_active_synapse.sum(axis=0) == 0)
                & (W_rec_active_synapse.sum(axis=1) == 0)
                & trainable_mask
            )

            # readout
            readout_free_neurons = (
                jnp.sum(
                    jnp.abs(
                        W_out_value
                        * self.modulation_activation(W_mask_value[cue_value, :, :])
                    )
                    > threshold,
                    axis=-1,
                )
                == 0
            )

            trainable_mask_value = recurrent_free_neurons & readout_free_neurons

            # jax output in backward order
            jax.debug.print(
                "===============trainable_neuron_size: {}", jnp.sum(trainable_mask)
            )
            jax.debug.print(
                "===============recurrent_free_neurons: {}",
                jnp.sum(recurrent_free_neurons),
            )
            jax.debug.print(
                "===============readout_free_neurons: {}", jnp.sum(readout_free_neurons)
            )
            jax.debug.print(
                "===============output trainable neuron size: {}",
                jnp.sum(trainable_mask_value),
            )

            # reset excitation of newly trainable/free neurons to 1.0
            excitation_value = jnp.where(trainable_mask_value, 1.0, excitation_value)

            # identity update
            task_index = jnp.max(identity)
            identity_value = jnp.where(trainable_mask_value, task_index + 1, identity)

            return (trainable_mask_value, excitation_value, identity_value)

        trainable_mask_value = self.trainable_mask.get_value()
        excitation_value = self.excitation.get_value()
        identity_value = self.identity.get_value()

        trainable_mask_value, excitation_value, identity_value = lax.cond(
            should_consolidate,
            lambda values: consolidate(*values),
            lambda values: values,
            (trainable_mask_value, excitation_value, identity_value),
        )

        self.cue[...] = cue_index
        self.trainable_mask[...] = trainable_mask_value
        self.excitation[...] = excitation_value
        self.identity[...] = identity_value

        # channel update
        self.channel[...] = lax.cond(
            is_new_cue,
            lambda channel: channel.at[cue_index].set(True),
            lambda _: _,
            channel_value,
        )

        # ==============================
        # 2. forward
        # ==============================
        return forward(
            self.rngs(),
            self.get_X_cue_TBC(X),
            X,
            self.W_in.get_value(),
            self.W_rec.get_value(),
            self.W_out.get_value(),
            self.b_h.get_value(),
            self.b_y.get_value(),
            self.excitation.get_value(),
            self.W_mask.get_value(),
            self.dt,
            self.tau,
            self.hidden_activation,
            self.modulation_activation,
            self.output_activation,
            self.noise_std,
        )

    def evaluation(self, X):
        return forward(
            self.rngs(),
            self.get_X_cue_TBC(X),
            X,
            self.W_in.get_value(),
            self.W_rec.get_value(),
            self.W_out.get_value(),
            self.b_h.get_value(),
            self.b_y.get_value(),
            self.excitation.get_value(),
            self.W_mask.get_value(),
            self.dt,
            self.tau,
            self.hidden_activation,
            self.modulation_activation,
            self.output_activation,
            self.noise_std,
        )
