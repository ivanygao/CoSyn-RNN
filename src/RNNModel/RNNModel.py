from flax import nnx
import matplotlib.pyplot as plt
import wandb
import numpy as np
import jax

from nntp.models.ModelBase import ModelBase, ModelStateBase
from nntp.models.enums import calculate_scale_by_distribution
from nntp.models.enums_getter import (
    get_optimizer,
    get_loss_function,
)
from nntp.models.ModelPlot import (
    plot_eigenvalue_spectrum,
    plot_matrices_heatmap_helper,
    plot_matrices_histogram_helper,
)
from .RNN import RNN


class ModelState(ModelStateBase):
    def __init__(self, model, optimizer):
        super().__init__(model)
        self.optimizer = optimizer  # nnx.Optimizer


class RNNModel(ModelBase):
    @staticmethod
    def get_model_cls() -> nnx.Module:
        return RNN

    @staticmethod
    def build_model_state(model, experiment_config: dict) -> ModelState:

        optimizer = nnx.Optimizer(
            model=model,
            tx=get_optimizer(
                experiment_config["optimizer"], experiment_config["learning_rate"]
            ),
            wrt=nnx.Param,
        )
        return ModelState(model=model, optimizer=optimizer)

    @staticmethod
    def build_training_step(experiment_config: dict) -> callable:
        loss_fn = get_loss_function(experiment_config["loss"])

        @nnx.jit
        def training_step(state, X, Y, M, epoch, batch_count, batch_length):

            def loss_function(model):
                predict = model(X)
                return loss_fn(predict, Y, M), predict

            (loss, predict), grads = nnx.value_and_grad(loss_function, has_aux=True)(
                state.model
            )
            state.optimizer.update(state.model, grads)

            return loss, predict, {}

        return training_step

    @staticmethod
    def build_evaluation_step(experiment_config: dict) -> callable:
        loss_fn = get_loss_function(experiment_config["loss"])

        @nnx.jit
        def evaluation_step(state, X, Y, M, epoch, batch_count, batch_length):
            predict = state.model.evaluation(X)
            loss = loss_fn(predict, Y, M)

            return loss, predict, {}

        return evaluation_step

    @staticmethod
    def build_plot_model_state(experiment_config: dict) -> callable:
        initial_weight_distribution = experiment_config["initial_weight_distribution"]
        scale = calculate_scale_by_distribution(
            initial_weight_distribution, experiment_config["hidden_size"]
        )

        def plot_model_state(model, path, epoch):
            # retrieve states
            W_in_value = model.W_in.get_value()
            W_rec_value = model.W_rec.get_value()
            W_out_value = model.W_out.get_value()
            b_h_value = model.b_h.get_value()
            b_y_value = model.b_y.get_value()

            # matrices_heatmap
            in_weight_heatmap = plot_matrices_heatmap_helper(
                W_in_value / scale,
                path,
                title=f"input_weight_heatmap_epoch_{epoch}",
            )
            rec_weight_heatmap = plot_matrices_heatmap_helper(
                W_rec_value / scale,
                path,
                title=f"recurrent_weight_heatmap_epoch_{epoch}",
            )
            out_weight_heatmap = plot_matrices_heatmap_helper(
                W_out_value / scale,
                path,
                title=f"output_weight_heatmap_epoch_{epoch}",
            )
            # matrices_histogram
            in_weights_histogram = plot_matrices_histogram_helper(
                {"W_in": W_in_value},
                path,
                f"input_weights_histogram_epoch_{epoch}",
            )
            rec_weights_histogram = plot_matrices_histogram_helper(
                {"W_rec": W_rec_value},
                path,
                f"recurrent_weights_histogram_epoch_{epoch}",
            )
            out_weights_histogram = plot_matrices_histogram_helper(
                {"W_out": W_out_value},
                path,
                f"output_weights_histogram_epoch_{epoch}",
            )
            bias_h_histogram = plot_matrices_histogram_helper(
                {"b_h": b_h_value},
                path,
                f"bias_h_histogram_epoch_{epoch}",
            )
            bias_y_histogram = plot_matrices_histogram_helper(
                {"b_y": b_y_value},
                path,
                f"bias_y_histogram_epoch_{epoch}",
            )

            # eigenvalue_spectrum
            fig = plot_eigenvalue_spectrum(W_rec_value)
            fig.savefig(path / f"eigenvalue_spectrum_epoch_{epoch}.png")
            eigenvalue_spectrum = wandb.Image(fig)
            plt.close(fig)

            return {
                "input_weight_heatmap": in_weight_heatmap,
                "rec_weight_heatmap": rec_weight_heatmap,
                "output_weight_heatmap": out_weight_heatmap,
                "input_weights_histogram": in_weights_histogram,
                "recurrent_weights_histogram": rec_weights_histogram,
                "output_weights_histogram": out_weights_histogram,
                "bias_h_histogram": bias_h_histogram,
                "bias_y_histogram": bias_y_histogram,
                "eigenvalue_spectrum": eigenvalue_spectrum,
            }

        return plot_model_state
