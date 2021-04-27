import numpy as np
from xgboost import XGBClassifier
import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def get_scores(y_test, ypred, ypred_proba, title: str, threshold_fpr: float = 0.001, figfile: str = None):
    print("{}: Accuracy: {}, F1-Micro: {}, F1-Macro: {}, ROC-AUC at 5e-3: {}"
          .format(title, accuracy_score(y_test, ypred),
                  f1_score(y_test, ypred, average="micro"), f1_score(
                      y_test, ypred, average="macro"),
                  roc_auc_score(y_true=y_test, y_score=ypred_proba, max_fpr=5e-3)))

    # Draw ROC curve
    fpr, tpr, thrx = roc_curve(y_test, ypred_proba)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 0.1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC:' + title)
    plt.legend(loc="lower right")
    if figfile is None:
        plt.show()
    else:
        plt.savefig(figfile, dpi=300)

    fprindex = np.where(fpr <= threshold_fpr)[0][-1]
    tpvalue = tpr[fprindex]
    print("TP: {} at FP:{}. thr:{}".format(tpvalue, fpr[fprindex], thrx[fprindex]))
    return thrx[fprindex]


def test_model(X_train: np.ndarray,
               y_train: np.ndarray,
               X_test: np.ndarray,
               y_test: np.ndarray,
               modelpath: str):

    clf: XGBClassifier = XGBClassifier()
    clf.load_model(fname=os.path.join(modelpath, "model_xgboost.dat"))

    ypred_test = clf.predict(X_test)
    ypred_test_proba = clf.predict_proba(X_test)[:, 1]
    get_scores(y_test=y_test, ypred=ypred_test, ypred_proba=ypred_test_proba, title="Test")

    ypred_train = clf.predict(X_train)
    ypred_train_proba = clf.predict_proba(X_train)[:, 1]
    get_scores(y_test=y_train, ypred=ypred_train, ypred_proba=ypred_train_proba, title="Train")


def predict_class(clf: XGBClassifier, samples: np.ndarray, threshold: float):
    return (clf.predict_proba(samples)[:, 1] > threshold).astype(np.int)
