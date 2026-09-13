from flax import nnx
import jax
import jax.numpy as jnp

from nntp.models.ModelBase import ModelBase, ModelStateBase
from nntp.models.enums import calculate_scale_by_distribution
from nntp.models.enums_getter import get_optimizer, get_loss_function
from nntp.models.ModelPlot import (
    plot_matrices_heatmap_helper,
    plot_task_block_sorted_by_threshold_count_heatmap_helper,
    plot_task_block_norm_heatmap_helper,
    plot_W_grid_helper,
)
from .CoSynRNNModelUtils import compute_W_rec_task_block_cross_norms

from .CoSynRNN import CoSynRNN

from RobertYang2019.RobertYang2019Evaluator import build_evaluator


class ModelState(ModelStateBase):
    def __init__(self, model, optimizer):
        super().__init__(model)
        self.optimizer = optimizer  # nnx.Optimizer


class CoSynRNNModel(ModelBase):
    @staticmethod
    def get_model_cls() -> nnx.Module:
        return CoSynRNN

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

        gain_sparsity_penalty = experiment_config["gain_sparsity_penalty"]
        recurrent_sparsity_penalty = experiment_config["recurrent_sparsity_penalty"]
        recurrent_incoming_penalty = experiment_config["recurrent_incoming_penalty"]
        recurrent_outgoing_penalty = experiment_config["recurrent_outgoing_penalty"]
        modulated_readout_penalty = experiment_config["modulated_readout_penalty"]

        evaluator = build_evaluator(experiment_config)

        angle_threshold = experiment_config["angle_threshold"]
        chance_level = (angle_threshold * 2) / 360

        @nnx.jit
        def training_step(state, X, Y, M, epoch, batch_count, batch_length):
            def loss_function(model):
                predict, W_out_modulated = model(
                    X, batch_count, epoch
                )  # W_out_modulated: [T, B, H, O]
                task_loss = loss_fn(predict, Y, M)

                eps = 1e-8
                H = model.hidden_size
                penalty_amplification = jax.lax.stop_gradient(
                    jnp.clip(
                        1.5
                        * (evaluator(predict, Y, M)["accuracy_angle"] - chance_level)
                        / (1.0 - chance_level),
                        0.0,
                        1.5,
                    )
                )

                # ==============================
                # Mask
                # ==============================
                trainable_mask_value_H = model.trainable_mask.get_value()
                trainable_mask_size = jnp.sum(trainable_mask_value_H)
                masked_TB = jnp.any(M != 0, axis=2)
                masked_TB_size = jnp.sum(masked_TB)

                # ==============================
                # loss
                # ==============================
                gain_value = model.gain_activation(model.gain.get_value())

                # ==============================
                # 1. gain: L1
                # ==============================
                gain_loss = (
                    penalty_amplification
                    * gain_sparsity_penalty
                    * (
                        jnp.sum(jnp.abs(gain_value * trainable_mask_value_H))
                        / trainable_mask_size
                    )
                )

                # ==============================
                # 2. W_out_modulated: L1L2
                # ==============================
                modulated_readout = jnp.sqrt(
                    jnp.sum((W_out_modulated) ** 2, axis=-1) + eps
                ) - jnp.sqrt(
                    eps
                )  # T, B, H
                modulated_readout_loss = (
                    penalty_amplification
                    * modulated_readout_penalty
                    * (
                        1 / model.scale
                    )  # adjust for scaling in W_out，assumed to be normal distribution initialization of W_out
                    * (
                        jnp.sum(modulated_readout * masked_TB[:, :, None])
                        / (masked_TB_size * H + eps)
                    )
                )

                # ==============================
                # 3.0 W_rec
                # ==============================
                W_rec_value = model.W_rec.get_value()
                W_rec_value_task = W_rec_value * trainable_mask_value_H[None, :]
                W2_rec_value_task = W_rec_value_task**2

                # ==============================
                # 3.1. W_rec: synapse-level L1 sparsity
                # ==============================
                recurrent_synapse_loss = (
                    penalty_amplification
                    * recurrent_sparsity_penalty
                    * jnp.sum(jnp.abs(W_rec_value_task))
                    / (H * trainable_mask_size + eps)
                )

                # ==============================
                # 3.2. W_rec: neuron-level L2 incoming norm
                # ==============================
                recurrent_incoming = jnp.sqrt(
                    jnp.sum(W2_rec_value_task, axis=0) + eps
                ) - jnp.sqrt(eps)
                recurrent_incoming_loss = (
                    penalty_amplification
                    * recurrent_incoming_penalty
                    * jnp.sum(recurrent_incoming)
                    / (trainable_mask_size + eps)
                )

                # ==============================
                # 2.3. W_rec: neuron-level L2 outgoing norm
                # ==============================
                recurrent_outgoing = jnp.sqrt(
                    jnp.sum(W2_rec_value_task, axis=1) + eps
                ) - jnp.sqrt(eps)
                recurrent_outgoing_loss = (
                    penalty_amplification
                    * recurrent_outgoing_penalty
                    * jnp.sum(recurrent_outgoing)
                    / H
                )

                sparsity_loss = (
                    gain_loss
                    + recurrent_synapse_loss
                    + recurrent_incoming_loss
                    + recurrent_outgoing_loss
                    + modulated_readout_loss
                )

                return task_loss + sparsity_loss, (
                    predict,
                    task_loss,
                    sparsity_loss,
                    penalty_amplification,
                )

            (
                loss,
                (predict, task_loss, sparsity_loss, penalty_amplification),
            ), grads = nnx.value_and_grad(loss_function, has_aux=True)(state.model)

            # ==============================
            # 1. Get current owned neurons
            # ==============================
            W_in_value = state.model.W_in.get_value()
            W_rec_value = state.model.W_rec.get_value()
            b_h_value = state.model.b_h.get_value()
            gain_value = state.model.gain.get_value()
            W_mask_value = state.model.W_mask.get_value()
            trainable_mask_value = state.model.trainable_mask.get_value()

            # ==============================
            # 2. Mask gradient
            # ==============================
            trainable_mask_value_2d = trainable_mask_value[None, :]
            grads["W_in"].value = grads["W_in"].value * trainable_mask_value_2d
            grads["W_rec"].value = grads["W_rec"].value * trainable_mask_value_2d
            grads["b_h"].value = grads["b_h"].value * trainable_mask_value
            grads["gain"].value = grads["gain"].value * trainable_mask_value
            trainable_W_mask_value = jax.nn.one_hot(
                state.model.get_cue_index(X), state.model.C, dtype=jnp.bool_
            )[:, None, None]
            grads["W_mask"].value = grads["W_mask"].value * trainable_W_mask_value

            # ==============================
            # 3. Optimizer update
            # ==============================
            state.optimizer.update(state.model, grads)

            # ==============================
            # 4. Hard mask state
            # ==============================
            W_in_new = state.model.W_in.get_value()
            W_rec_new = state.model.W_rec.get_value()
            b_h_new = state.model.b_h.get_value()
            W_mask_new = state.model.W_mask.get_value()
            gain_new = state.model.gain.get_value()

            state.model.W_in[...] = jnp.where(
                trainable_mask_value_2d, W_in_new, W_in_value
            )
            state.model.W_rec[...] = jnp.where(
                trainable_mask_value_2d, W_rec_new, W_rec_value
            )
            state.model.b_h[...] = jnp.where(trainable_mask_value, b_h_new, b_h_value)
            state.model.gain[...] = jnp.where(
                trainable_mask_value, gain_new, gain_value
            )
            state.model.W_mask[...] = jnp.where(
                trainable_W_mask_value, W_mask_new, W_mask_value
            )

            # ------------------------------
            # Metrics
            # ------------------------------
            def threshold_stats(value_abs, valid_mask, threshold):
                valid_mask = jnp.broadcast_to(valid_mask, value_abs.shape)

                valid_count = jnp.sum(valid_mask)
                valid_count = jnp.maximum(valid_count, 1)

                le_mask = (value_abs <= threshold) & valid_mask

                le_count = jnp.sum(le_mask)
                le_fraction = le_count / valid_count

                return le_fraction

            trainable_mask_value = state.model.trainable_mask.get_value()  # [H]

            gain_value_raw = state.model.gain.get_value()  # [H]
            gain_value = jax.nn.relu(gain_value_raw)  # [H]

            # ------------------------------
            # gain threshold metrics
            # ------------------------------
            gain_value_abs = jnp.abs(gain_value)
            gain_le_0_fraction = threshold_stats(
                gain_value_abs, trainable_mask_value, 0.0
            )

            # ------------------------------
            # W_rec threshold metrics
            # ------------------------------
            W_rec_value = state.model.W_rec.get_value()  # [H, H]
            W_rec_value_abs = jnp.abs(W_rec_value)

            # valid incoming-to-trainable columns
            valid_rec_mask = trainable_mask_value[None, :]  # [1, H]

            W_rec_le_1e4_fraction = threshold_stats(
                W_rec_value_abs, valid_rec_mask, 1e-4
            )
            W_rec_le_1e5_fraction = threshold_stats(
                W_rec_value_abs, valid_rec_mask, 1e-5
            )

            # ------------------------------
            # W_out modulated threshold metrics
            # ------------------------------
            cue_index = state.model.get_cue_index(X)
            modulation = state.model.modulation_activation(
                state.model.W_mask.get_value()[cue_index, :, :]
            )  # [H, O]
            W_out_modulated_abs = jnp.abs(
                state.model.W_out.get_value() * modulation
            )  # [H, O]

            valid_modulation_mask = jnp.ones_like(
                W_out_modulated_abs, dtype=bool
            )  # [H, O]

            W_out_modulated_le_1e4_fraction = threshold_stats(
                W_out_modulated_abs, valid_modulation_mask, 1e-4
            )
            W_out_modulated_le_1e5_fraction = threshold_stats(
                W_out_modulated_abs, valid_modulation_mask, 1e-5
            )

            return (
                loss,
                predict,
                {
                    "loss/task": task_loss,
                    "loss/sparsity": sparsity_loss,
                    "penalty_amplification": penalty_amplification,
                    "L1/exci_masked": jnp.mean(
                        gain_value_abs * trainable_mask_value
                    ),
                    "gain_le_0_fraction": gain_le_0_fraction,
                    "W_rec_le_1e-4_fraction": W_rec_le_1e4_fraction,
                    "W_rec_le_1e-5_fraction": W_rec_le_1e5_fraction,
                    "W_out_modulated_le_1e-4_fraction": W_out_modulated_le_1e4_fraction,
                    "W_out_modulated_le_1e-5_fraction": W_out_modulated_le_1e5_fraction,
                    "number_of_trainable_neuron": jnp.sum(trainable_mask_value),
                },
            )

        return training_step

    @staticmethod
    def build_evaluation_step(experiment_config: dict) -> callable:
        loss_fn = get_loss_function(experiment_config["loss"])

        @nnx.jit
        def evaluation_step(state, X, Y, M, epoch, batch_count, batch_length):
            predict, _ = state.model.evaluation(X)
            loss = loss_fn(predict, Y, M)

            return (loss, predict, {})

        return evaluation_step

    @staticmethod
    def build_plot_model_state(experiment_config: dict) -> callable:
        threshold = experiment_config["threshold"]
        initial_weight_distribution = experiment_config["initial_weight_distribution"]
        scale = calculate_scale_by_distribution(
            initial_weight_distribution, experiment_config["hidden_size"]
        )

        def plot_model_state(model, path, epoch):
            # retrieve states
            identity_value = model.identity.get_value()
            channel_value = model.channel.get_value()

            gain = jax.nn.relu(model.gain.get_value())
            gain_heatmap = plot_matrices_heatmap_helper(
                gain[:, None].T, path, f"gain_heatmap_epoch_{epoch}"
            )

            W_rec_value = model.W_rec.get_value()
            W_rec_value_scaled = W_rec_value / scale

            # ==============================================
            # rec_task_block_norm_heatmap
            # ==============================================
            rec_task_block_norm_heatmap = plot_task_block_norm_heatmap_helper(
                compute_W_rec_task_block_cross_norms(
                    (W_rec_value_scaled).T, identity_value, "fro"
                ),
                f"rec_task_block_norm_heatmap_epoch_{epoch}",
                path,
            )

            # ==============================================
            # rec_sorted_by_threshold_count_clip_heatmap
            # ==============================================
            rec_sorted_by_threshold_count_clip_heatmap = (
                plot_task_block_sorted_by_threshold_count_heatmap_helper(
                    W_rec_value.T,
                    identity_value,
                    threshold,
                    path,
                    f"rec_sorted_by_threshold_count_clip_heatmap_epoch_{epoch}",
                )
            )

            # ==============================================
            # W_out_modulated_grid_clip_heatmap
            # ==============================================
            W_out_modulated_grid_clip_heatmap = plot_W_grid_helper(
                model.W_out[None, :, :]
                * model.modulation_activation(model.W_mask.get_value()),
                channel_value,
                threshold,
                path,
                f"W_out_modulated_grid_clip_heatmap_epoch_{epoch}",
            )

            return {
                "gain_heatmap": gain_heatmap,
                "rec_task_block_norm_heatmap": rec_task_block_norm_heatmap,
                "rec_sorted_by_threshold_count_clip_heatmap": rec_sorted_by_threshold_count_clip_heatmap,
                "W_out_modulated_grid_clip_heatmap": W_out_modulated_grid_clip_heatmap,
            }

        return plot_model_state
