from dataclasses import dataclass
from pydantic import BaseModel


@dataclass(frozen=True)
class Experiment:
    name: str
    seed: int
    model_path: str
    debug: bool
    cfg: BaseModel

    def __str__(self) -> str:
        lines = []
        lines.append(f"Experiment   : {self.name}")
        lines.append(f"Seed         : {self.seed}")
        lines.append(f"Model        : {self.model_path}")
        lines.append(f"Debug        : {self.debug}")
        lines.append(f"Configs      : {self.cfg}")
        return "\n".join(lines)
