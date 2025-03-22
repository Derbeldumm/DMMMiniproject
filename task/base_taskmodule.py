from abc import abstractmethod, ABC
from discopy.grammar.pregroup import Diagram

class base_taskmodule(ABC):
    """A base module for task sepcific parts"""

    @abstractmethod
    def get_scenarios(self) -> tuple[list[Diagram], list[bool]]:
        pass

    @abstractmethod
    def get_hints(self) -> list[Diagram]:
        pass

    @abstractmethod
    def get_dictionary(self) -> dict[str, Diagram]:
        pass

    @abstractmethod
    def get_type_strings(self) -> list[str]:
        pass