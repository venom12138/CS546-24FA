from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict

class BaseEnv(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the environment.
        """
        pass

    @abstractmethod
    def reset(self) -> Any:
        """
        Reset the environment to an initial state and return the initial observation.
        :return: The initial observation.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.
        :param action: An action provided by the agent.
        :return: A tuple (observation, reward, done, info).
        """
        pass

    @abstractmethod
    def render(self, mode: str = 'human') -> Any:
        """
        Render the environment.
        :param mode: The mode in which to render the environment.
        :return: Rendered image or None.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass
