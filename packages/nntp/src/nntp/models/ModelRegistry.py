from enum import Enum
from pydantic import BaseModel
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    source: str
    description: str
    config: BaseModel
    model_path: str

    def __str__(self) -> str:
        lines = []
        lines.append(f"Model      : {self.key}")
        lines.append(f"Name         : {self.name}")
        lines.append(f"Source       : {self.source}")
        lines.append(f"Description  : {self.description}")
        return "\n".join(lines)


class ModelRegistry:
    _registry: dict[str, Model] = {}

    @classmethod
    def register(
        cls,
        key: str,
        name: str,
        source: str,
        description: str,
        config: BaseModel,
        model_path: str,
    ) -> None:
        cls._registry[key] = Model(
            key=key,
            name=name,
            source=source,
            description=description,
            config=config,
            model_path=model_path,
        )

    @classmethod
    def get(cls, key: str) -> Model:
        return cls._registry[key]

    @classmethod
    def export_registry(cls) -> Enum:
        return Enum("MODELS", cls._registry)

    @classmethod
    def export_registry_key(cls) -> Enum:
        return Enum("MODELS_KEY", {k: k for k in cls._registry}, type=str)

    @classmethod
    def export_registry_config(cls) -> Enum:
        return Enum("MODELS_CONFIG", {k: v.config for k, v in cls._registry.items()})
