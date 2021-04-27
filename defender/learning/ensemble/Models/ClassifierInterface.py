from abc import ABC, abstractmethod
import typing
from .ClassificationResult import ClassificationResult


class ClassifierInterface(ABC):

    @abstractmethod
    def predict(self, bytez, pe_binary=None) -> list:
        pass

    @abstractmethod
    def predict_proba(self, bytez, pe_binary=None) -> list:
        pass

    @abstractmethod
    def predict_all_proba(self, bytez, pe_binary=None) -> typing.List[ClassificationResult]:
        pass
