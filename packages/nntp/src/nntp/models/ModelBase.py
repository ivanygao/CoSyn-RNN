from abc import ABC, abstractmethod
from flax import nnx


class ModelStateBase(nnx.Module):
    def __init__(self, model):
        self.model = model  # nnx.Module


# Generator interface, all generator must implement this
class ModelBase(ABC):
    """
    Interface for all project.
    """

    @staticmethod
    @abstractmethod
    def get_model_cls() -> nnx.Module:
        """
        Should return a nnx.Module subclass (not instance)
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_model_state(model: nnx.Module, cfg: dict) -> ModelStateBase:
        """
        Return a ModelState instance that includes a model and at least one optimizer
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_training_step(cfg: dict) -> callable:
        """
        Return a training_step(state, X, Y, M) function which return value is (loss, predict)
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_evaluation_step(cfg: dict) -> callable:
        """
        Return an evaluation_step(state, X, Y, M) function which return value is (loss, predict)
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_plot_model_state(experiment_config: dict) -> callable:
        """
        Return an plot_model_state(model, path, epoch) function which return value is a dictionary of info for wandb to upload.
        """
        raise NotImplementedError
