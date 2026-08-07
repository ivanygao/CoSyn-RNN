from pydantic import BaseModel, Field

from nntp.models.enums import Activation, Distribution


class CoSynRNNConfig(BaseModel):
    dt: list[float] = Field(min_length=1)
    tau: list[float] = Field(min_length=1)
    hidden_size: list[int] = Field(min_length=1, min=1)
    hidden_activation: list[Activation] = Field(min_length=1)
    excitation_activation: list[Activation] = Field(min_length=1)
    modulation_activation: list[Activation] = Field(min_length=1)
    output_activation: list[Activation] = Field(min_length=1)
    initial_weight_distribution: list[Distribution] = Field(min_length=1)
    initial_spectral_radius: list[float] = Field(min_length=1)
    noise_std: list[float] = Field(min_length=1)

    cue_index_start: list[int] = Field(min_length=1)
    threshold: list[float] = Field(min_length=1)
