from pydantic import BaseModel, Field

from nntp.models.ModelRegistry import ModelRegistry
from nntp.models.enums import Optimizer, Loss
from .RNNConfig import RNNConfig, PositiveFloat, NonNegativeFloat


class TrainableParams(BaseModel):
    W_in: bool
    W_rec: bool
    W_out: bool
    b_h: bool
    b_y: bool


class RNNModelConfig(RNNConfig):
    trainable_params: list[TrainableParams] = Field(min_length=1)

    loss: list[Loss] = Field(min_length=1)
    optimizer: list[Optimizer] = Field(min_length=1)
    learning_rate: list[PositiveFloat] = Field(min_length=1)

    fixation_tolerance: list[NonNegativeFloat] = Field(min_length=1)
    radius_tolerance: list[NonNegativeFloat] = Field(min_length=1)
    angle_tolerance: list[NonNegativeFloat] = Field(min_length=1)
    fixation_threshold: list[NonNegativeFloat] = Field(min_length=1)
    radius_threshold: list[NonNegativeFloat] = Field(min_length=1)
    angle_threshold: list[NonNegativeFloat] = Field(min_length=1)


ModelRegistry.register(
    "RNNModel",
    "Example RNN Model",
    "ivanygao",
    "A vanilla RNN model example",
    RNNModelConfig,
    "RNNModel.RNNModel:RNNModel",
)
