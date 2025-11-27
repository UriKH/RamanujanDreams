from abc import ABC, abstractmethod
from ramanujantools import Position
from typing import List, Optional, Tuple
import sympy as sp


class Searchable(ABC):
    @abstractmethod
    def in_space(self, point: Position) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def calc_delta(self, start: Position, trajectory: Position) -> float:
        raise NotImplementedError()

    @abstractmethod
    def get_interior_point(self) -> Position:
        raise NotImplementedError()

    @abstractmethod
    def sample_trajectories(self, n_samples: int) -> List[Position]:
        raise NotImplementedError()

