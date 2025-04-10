from abc import abstractmethod, ABC
from discopy.grammar.pregroup import Diagram

class base_taskmodule(ABC):
    """A base module for task sepcific parts"""

    @abstractmethod
    def get_scenarios(self) -> tuple[list[Diagram], list[bool]]:
        pass

    @abstractmethod
    def get_hints(self) -> tuple[list[Diagram], list[bool]]:
        pass

    @abstractmethod
    def get_gates_to_analyse(self) -> list[Diagram]:
        pass
    