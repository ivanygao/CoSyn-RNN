from enum import Enum
from dataclasses import dataclass


@dataclass
class Task:
    key: str
    name: str
    source: str
    description: str
    generator: callable
    decoder: str
    evaluator: str

    def __str__(self) -> str:
        lines = []
        lines.append(f"Task         : {self.key}")
        lines.append(f"Name         : {self.name}")
        lines.append(f"Source       : {self.source}")
        lines.append(f"Description  : {self.description}")
        lines.append(f"Generator  : {self.generator}")
        lines.append(f"Decoder  : {self.decoder}")
        lines.append(f"Evaluator  : {self.evaluator}")
        return "\n".join(lines)


class TaskRegistry:
    _registry: dict[str, Task] = {}

    @classmethod
    def register(
        cls,
        key: str,
        name: str,
        source: str,
        description: str,
        generator: callable,
        decoder_path: str,
        evaluator_path: str,
    ) -> None:
        cls._registry[key] = Task(
            key=key,
            name=name,
            source=source,
            description=description,
            generator=generator,
            decoder=decoder_path,
            evaluator=evaluator_path,
        )

    @classmethod
    def get(cls, key: str) -> Task:
        return cls._registry[key]

    @classmethod
    def export_registry(cls) -> Enum:
        return Enum("TASKS", cls._registry)

    @classmethod
    def export_registry_key(cls) -> Enum:
        return Enum("TASKS_KEY", {k: k for k in cls._registry.keys()}, type=str)
