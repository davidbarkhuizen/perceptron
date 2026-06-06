from abc import ABC, abstractmethod


class AbstractNode(ABC):
    @abstractmethod
    def value(self) -> float:
        raise NotImplementedError()
