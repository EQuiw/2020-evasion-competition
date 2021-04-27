import sys
from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from learning.emberboost.ember.boost.train_boost_utils import baseparams
from sklearn.utils.random import sample_without_replacement
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, make_scorer)
import os
import typing


def train(X_train: np.ndarray,
          y_train: np.ndarray,
          usemonotonic: bool,
          resultsdir: str,
          bo_iters: int, bo_init_points: int, noofthreads: int = 10,
          gpu_id: typing.Optional[int] = None):
    """
    Train with xgboost.
    """
    noofsamples = None

    # A. Preprocess data
    if noofsamples is not None:
        print("Warning. I just use a small train set for debugging", file=sys.stderr)
        xpos = sample_without_replacement(X_train.shape[0], noofsamples, random_state=31)
        xpos = np.sort(xpos)
        X_train = X_train[xpos]
        y_train = y_train[xpos]
    print("Samples per class:", np.bincount(y_train.astype(np.int)))

    nooffeatures = X_train.shape[1]


    # B. Make bounded AUC scoring function
    scorerboundedauc = make_scorer(roc_auc_score, max_fpr=5e-3)


    # C. Time-based Cross-Validation with Bayesian Optimization
    def xgb_evaluate(max_depth, gamma, colsample_bytree, min_child_weight, num_boost_rounds, reg_lambda, subsample):
        params_constrained = baseparams(usemonotonic=usemonotonic, nooffeatures=nooffeatures, gpu_id=gpu_id)
        params_constrained.update({
            'max_depth': int(max_depth),
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
            'reg_lambda': reg_lambda
        })

        clf = XGBClassifier(n_estimators=int(num_boost_rounds), nthread=noofthreads)# , nthread=1)
        clf.get_xgb_params()
        clf = clf.set_params(**params_constrained)

        progressive_cv = TimeSeriesSplit(n_splits=3).split(X_train)
        scores = cross_val_score(clf, X_train, y_train, cv = progressive_cv, scoring=scorerboundedauc)

        # Bayesian optimization only knows how to maximize, not minimize, but here we can return bounded auc value
        return np.mean(scores)

    # to do: increase max_bin if monotonic?
    xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 10),
                                                 'gamma': (0.025, 1),
                                                 # 'eta': (0.1, 0.3),  # to do
                                                 'subsample': (0.5, 0.8),
                                                 'reg_lambda': (5, 7.5),
                                                 'min_child_weight': (0, 5),
                                                 'colsample_bytree': (0.3, 0.8),
                                                 'num_boost_rounds': (5, 500)})

    # Use the expected improvement acquisition function to handle negative numbers
    # Optimally needs quite a few more initiation points and number of iterations
    xgb_bo.maximize(init_points=bo_init_points, n_iter=bo_iters, acq='ei')


    # D. Train final model
    params_constrained = baseparams(usemonotonic=usemonotonic, nooffeatures=nooffeatures, gpu_id=gpu_id)
    params_constrained.update(xgb_bo.max['params'])

    # need to convert max-depth to int
    params_constrained['max_depth'] = int(params_constrained['max_depth'])
    num_boost_rounds = int(params_constrained['num_boost_rounds'])
    del params_constrained['num_boost_rounds']

    # Train a new model with the best parameters from the search
    # We'll use XGBClassifier here
    clffinal: XGBClassifier = XGBClassifier(n_estimators=num_boost_rounds, nthread=noofthreads)
    clffinal = clffinal.set_params(**params_constrained)
    clffinal.fit(X=X_train, y=y_train)

    # E. Save
    clffinal.save_model(fname=os.path.join(resultsdir, "model_xgboost.dat"))