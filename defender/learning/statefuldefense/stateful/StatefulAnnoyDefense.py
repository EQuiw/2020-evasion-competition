import random
from annoy import AnnoyIndex  # pip install --user annoy
import numpy as np
import typing

from learning.statefuldefense.stateful.StatefulDefense import StatefulDefense


class StatefulAnnoyDefense(StatefulDefense):
    """
    Stateful defense, adapted from given defense in contest.
    """

    ADV_INDEX_SIZE = 512

    def __init__(self,
                 seed: int,
                 verbose: bool = False,
                 ball_thresh: float = 0.25,   # threshold for L1 distance to previously-seen malware
                 max_history: int = 10_000,  # keep up to this much query history
                 ):

        super().__init__(seed=seed, max_history=max_history, verbose=verbose)
        self.ball_thresh = ball_thresh

    def _check(self, score: float, ismalware: bool, features: np.ndarray, bytez, pe_binary) \
            -> typing.Tuple[bool, dict]:

        assert features.shape[0] == 1
        features = features[0, :self.ADV_INDEX_SIZE]

        if ismalware:
            self.malicious_queries.append((features, score))
            # if the list is too big, shuffle and trim (keep a random subset)
            while len(self.malicious_queries) > self.max_history:
                # remove a random element
                self.malicious_queries.pop(index=random.randint(0, len(self.malicious_queries)))

        elif len(self.malicious_queries) > 0:
            # is it sufficiently similar to some malicious sample I've seen previously?
            t = AnnoyIndex(self.ADV_INDEX_SIZE, 'manhattan')
            for i, (m, _) in enumerate(self.malicious_queries):
                t.add_item(i, m)
            t.build(20)

            # is the core of the is file similar to a malicious file I've seen?
            q = features
            nn_ix = t.get_nns_by_vector(q, 10)

            dists = [np.linalg.norm(self.malicious_queries[ix][0] - q, 1) for ix in nn_ix]

            assert dists[0] == min(dists)  # Actually no min necessary.
            # min_dist = min(dists)  # how close is the actual nearest neighbor?
            min_dist = dists[0]

            if min_dist < self.ball_thresh:
                if self.verbose is True:
                    print("Detected Adversarial Example!", 1-min_dist, self.ball_thresh)
                return True, {'dist': min_dist}
            else:
                return False, {'dist': min_dist, 'dists': dists}

        return False, {}

