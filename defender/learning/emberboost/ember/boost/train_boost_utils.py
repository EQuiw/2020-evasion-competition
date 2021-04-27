import typing
import learning.emberboost.ember as ember

def create_mono(nooffeatures):
    st = "(1,"
    for i in range(1,nooffeatures-1):
        st = st+"1,"
    st += "1)"
    return st


def baseparams(usemonotonic: bool, nooffeatures: int, gpu_id=typing.Optional[int]):
    a = {
        'tree_method' : ["exact", "hist", "gpu_hist"][1], # <- for monotonic, we should use exact, hist or gpu_hist.
        # 'subsample': 0.8,
        'eta': 0.15,
        'objective' : 'binary:logistic', # hinge
    }
    if usemonotonic is True:
        a['monotone_constraints'] = create_mono(nooffeatures)

    if gpu_id is not None:
        assert type(gpu_id) == int
        a['gpu_id'] = gpu_id
        a['tree_method'] = "gpu_hist"

    return a


def read_ember_features_from_disk(emberpath, feature_version, extractor):
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir=emberpath,
                                                                      feature_version=feature_version,
                                                                      extractor=extractor)
    train_rows = (y_train != -1)
    X_train = X_train[train_rows]
    y_train = y_train[train_rows]
    return X_train, y_train, X_test, y_test