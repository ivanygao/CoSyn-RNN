from pydantic import Field
from typing import Annotated

from nntp.models.ModelRegistry import ModelRegistry
from nntp.models.enums import Optimizer, Loss

from .CoSynRNNConfig import CoSynRNNConfig

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class CoSynRNNModelConfig(CoSynRNNConfig):
    loss: list[Loss] = Field(min_length=1)
    optimizer: list[Optimizer] = Field(min_length=1)
    learning_rate: list[PositiveFloat] = Field(min_length=1)

    gain_sparsity_penalty: list[NonNegativeFloat] = Field(min_length=1)
    recurrent_sparsity_penalty: list[NonNegativeFloat] = Field(min_length=1)
    recurrent_incoming_penalty: list[NonNegativeFloat] = Field(min_length=1)
    recurrent_outgoing_penalty: list[NonNegativeFloat] = Field(min_length=1)
    modulated_readout_penalty: list[NonNegativeFloat] = Field(min_length=1)

    fixation_tolerance: list[NonNegativeFloat] = Field(min_length=1)
    radius_tolerance: list[NonNegativeFloat] = Field(min_length=1)
    angle_tolerance: list[NonNegativeFloat] = Field(min_length=1)
    fixation_threshold: list[NonNegativeFloat] = Field(min_length=1)
    radius_threshold: list[NonNegativeFloat] = Field(min_length=1)
    angle_threshold: list[NonNegativeFloat] = Field(min_length=1)


ModelRegistry.register(
    "CoSynRNNModel",
    "Context-modulated Synaptic-states Recurrent Neural Network Model",
    "Yuan Gao, Stefan Mihalas, Denis Turcu",
    "Context-modulated Synaptic-states Recurrent Neural Network Model",
    CoSynRNNModelConfig,
    "CoSynRNNModel.CoSynRNNModel:CoSynRNNModel",
)
