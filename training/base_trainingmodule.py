from abc import abstractmethod, ABC
from task.base_taskmodule import base_taskmodule


class base_trainingmodule(ABC):
    """A base module for training sepcific parts"""

    @abstractmethod
    def learn_meanings(self, task_module: base_taskmodule) -> dict:
        pass
    
    