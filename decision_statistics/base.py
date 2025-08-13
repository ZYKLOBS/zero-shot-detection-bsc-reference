from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List


class BaseDecider(ABC):
    """Abstract interface for decision metrics (deciders)"""

    def __init__(self):
        #Add name here as attribute
        self.name = "you called the abstract decider, this should not happen"

    @abstractmethod
    def get_D(self, nll: List, entropy: List) -> List:
        """Returns coding cost gap D =  NLL - H , see ZED paper for details https://arxiv.org/pdf/2409.15875
        This method should implement spatial averaging according to ZED paper!"""
        pass

    @abstractmethod
    def get_delta(self, D: List) -> List:
        """Returns difference (delta) between D^(n) and D^(n+1) for multiple Resolution cases"""
        pass

    #If you can think of other interesting metrics you could add them here

