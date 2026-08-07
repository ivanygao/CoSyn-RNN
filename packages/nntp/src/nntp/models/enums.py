from enum import Enum
import numpy as np


class Activation(str, Enum):
    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SIGMOID5 = "sigmoid5"
    SIGMOID10 = "sigmoid10"
    SIGMOID15 = "sigmoid15"
    SIGMOID20 = "sigmoid20"
    SIGMOID25 = "sigmoid25"
    SIGMOID30 = "sigmoid30"
    SIGMOID40 = "sigmoid40"
    SIGMOID50 = "sigmoid50"
    SIGMOID75 = "sigmoid75"
    SIGMOID100 = "sigmoid100"
    LINEAR = "linear"
    RECTANH = "rectanh"
    LEAKYRELU = "leakyrelu"
    LEAKYRELULEAKY = "leakyreluleaky"


class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


class Loss(str, Enum):
    MSE = "mse"
    ASE = "ase"


class Distribution(str, Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"


def calculate_scale_by_distribution(distribution: Distribution, shape: int):
    if distribution == Distribution.NORMAL:
        return np.sqrt(1 / shape)
    elif distribution == Distribution.UNIFORM:
        return np.sqrt(3 / shape)
    else:
        return 1
