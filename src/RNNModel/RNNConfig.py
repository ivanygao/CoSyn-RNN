from pydantic import BaseModel, Field
from typing import Annotated

from nntp.models.enums import Activation, Distribution

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class RNNConfig(BaseModel):
    dt: list[PositiveFloat] = Field(min_length=1)
    tau: list[PositiveFloat] = Field(min_length=1)
    hidden_size: list[PositiveInt] = Field(min_length=1, min=1)
    hidden_activation: list[Activation] = Field(min_length=1)
    output_activation: list[Activation] = Field(min_length=1)
    initial_weight_distribution: list[Distribution] = Field(min_length=1)
    initial_spectral_radius: list[NonNegativeFloat] = Field(min_length=1)
    noise_std: list[NonNegativeFloat] = Field(min_length=1)
