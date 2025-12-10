import enum
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np

class Handedness(enum.Enum):
    LEFT = "Left"
    RIGHT = "Right"

class NormalizedPoint:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

class HandPoseDetector(ABC):
    @abstractmethod
    def detect(self, img: np.ndarray) -> Optional[Tuple[list[NormalizedPoint], Handedness]]:
        pass
