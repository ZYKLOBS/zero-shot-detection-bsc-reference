from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List


class BaseLoader(ABC):
    """Abstract interface for all data loaders."""

    def __init__(self):
        #Add name here as attribute
        self.name = "you called the abstract loader, this should not happen"

    @abstractmethod
    def get_nll(self, file: Path) -> List:
        """Return the NLL data for a single file."""
        pass

    @abstractmethod
    def get_entropy(self, file: Path) -> List:
        """Return the entropy data for a single file."""
        pass

