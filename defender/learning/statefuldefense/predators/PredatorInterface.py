from abc import ABC, abstractmethod


class PredatorInterface(ABC):
    """
    Scanning files for anomalies.
    """

    @abstractmethod
    def check_file(self, bytez, lief_binary) -> bool:
        pass
