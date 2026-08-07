import jax.numpy as jnp
import optax
from jax import nn

from .enums import Activation, Optimizer, Loss, Distribution


def leaky_relu_leaky(x, alpha=0.01):
    return jnp.where(
        x <= 1.0,
        nn.leaky_relu(x, negative_slope=alpha),
        1.0 + alpha * (x - 1.0),
    )


def get_activation_function(name: Activation):
    match name:
        case Activation.TANH:
            return jnp.tanh
        case Activation.RELU:
            return nn.relu
        case Activation.SIGMOID:
            return nn.sigmoid
        case Activation.SIGMOID5:
            return lambda x: nn.sigmoid(5.0 * x)
        case Activation.SIGMOID10:
            return lambda x: nn.sigmoid(10.0 * x)
        case Activation.SIGMOID15:
            return lambda x: nn.sigmoid(15.0 * x)
        case Activation.SIGMOID20:
            return lambda x: nn.sigmoid(20.0 * x)
        case Activation.SIGMOID25:
            return lambda x: nn.sigmoid(25.0 * x)
        case Activation.SIGMOID30:
            return lambda x: nn.sigmoid(30.0 * x)
        case Activation.SIGMOID40:
            return lambda x: nn.sigmoid(40.0 * x)
        case Activation.SIGMOID50:
            return lambda x: nn.sigmoid(50.0 * x)
        case Activation.SIGMOID75:
            return lambda x: nn.sigmoid(75.0 * x)
        case Activation.SIGMOID100:
            return lambda x: nn.sigmoid(100.0 * x)
        case Activation.LINEAR:
            return nn.identity
        case Activation.RECTANH:
            return lambda x: nn.relu(jnp.tanh(x))
        case Activation.LEAKYRELU:
            return nn.leaky_relu
        case Activation.LEAKYRELULEAKY:
            return leaky_relu_leaky
        case _:
            raise ValueError(f"Unsupported activation function: {name}")


def get_optimizer(name: Optimizer, learning_rate: float):
    match name:
        case Optimizer.ADAM:
            return optax.adam(learning_rate)
        case Optimizer.SGD:
            return optax.sgd(learning_rate)
        case _:
            raise ValueError(f"Unsupported optimizer: {name}")


def get_loss_function(name: Loss):
    match name:
        case Loss.MSE:

            def mse(preds, targets, mask):
                numerator = jnp.sum((preds - targets) ** 2 * mask, axis=(0, 2))
                denominator = jnp.sum(mask, axis=(0, 2))
                valid = denominator > 0  # remove padding trial
                numerator = jnp.where(valid, numerator, 0.0)
                denominator = jnp.where(valid, denominator, 1.0)
                return jnp.sum(numerator / denominator) / jnp.maximum(jnp.sum(valid), 1)

            return mse
        case Loss.ASE:

            def ase(preds, targets, mask):
                numerator = jnp.sum(jnp.abs(preds - targets) * mask, axis=(0, 2))
                denominator = jnp.sum(mask, axis=(0, 2))
                valid = denominator > 0  # remove padding trial
                numerator = jnp.where(valid, numerator, 0.0)
                denominator = jnp.where(valid, denominator, 1.0)
                return jnp.sum(numerator / denominator) / jnp.maximum(jnp.sum(valid), 1)

            return ase
        case _:
            raise ValueError(f"Unsupported loss function: {name}")
