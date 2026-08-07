from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class Strategy(str, Enum):
    SEQUENTIAL = "sequential"
    SHUFFLE = "shuffle"


class BatchMode(str, Enum):
    AUTO = "auto"
    FIXED = "fixed"


# =====================================================
# Logger
# =====================================================
class LoggerConfig(BaseModel):
    checkpoint_interval: int = 1
    wandb: Optional[str] = None


# =====================================================
# Dataset
# =====================================================
class DatasetConfig(BaseModel):
    training_size: int
    validation_size: int
    test_size: int


# =====================================================
# Training
# =====================================================
class TrainingConfig(BaseModel):
    batch_mode: BatchMode
    batch_size: Optional[int] = None


def create_root_config(
    TASKS_KEY: type[Enum],
    MODELS_KEY: type[Enum],
    MODELS_CONFIG: type[Enum],
) -> type[BaseModel]:
    """
    Create RootConfig after all user tasks and models
    have been registered.
    """

    # =====================================================
    # Workflow
    # =====================================================
    class WorkflowConfig(BaseModel):
        name: str = Field(min_length=1)
        epochs: int = Field(gt=0)
        training_strategy: Strategy

        training_tasks: list[TASKS_KEY] = Field(min_length=1)
        validation_tasks: list[TASKS_KEY] = Field(min_length=1)
        test_tasks: list[TASKS_KEY] = Field(min_length=1)

    # =====================================================
    # RootConfig
    # =====================================================
    class RootConfig(BaseModel):
        name: str = Field(min_length=1)
        note: str
        description: str = Field(min_length=1)

        seed: int | list[int]

        logger: LoggerConfig
        dataset: DatasetConfig
        training: TrainingConfig
        workflow: list[WorkflowConfig] = Field(min_length=1)

        models: dict[MODELS_KEY, Any]

        @model_validator(mode="after")
        def validate_models(self):
            parsed = {}

            for name, cfg in self.models.items():
                config_class = MODELS_CONFIG[name.value].value
                parsed[name] = config_class(**cfg)

            self.models = parsed
            return self

    return RootConfig
