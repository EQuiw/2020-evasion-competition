import random
import numpy as np
from abc import ABC, abstractmethod
import typing
import sys


class StatefulDefense(ABC):

    def __init__(self,
                 seed: int,
                 max_history: int,
                 verbose: bool
                 ):

        random.seed(seed)
        assert max_history >= 1
        self.max_history = max_history
        self.malicious_queries = []
        self.verbose = verbose

    def check(self, score: float, ismalware: bool, features: np.ndarray, bytez, pe_binary) -> typing.Tuple[bool, dict]:
        """
        Returns True if attack was detected.
        And a dictionary, that can contain some debug information (such as raw scores, and so on).
        """
        try:
            assert features.shape[0] == 1
            score_dict = self._check(score=score, ismalware=ismalware, features=features, bytez=bytez, pe_binary=pe_binary)
            return score_dict
        except Exception as e:
            print("Exception in stateful defense: {}".format(str(e)), file=sys.stderr)
            return False, {}

    @abstractmethod
    def _check(self, score: float, ismalware: bool, features: np.ndarray, bytez, pe_binary) \
            -> typing.Tuple[bool, dict]:
        pass
