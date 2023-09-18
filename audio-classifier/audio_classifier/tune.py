from .utils import get_data
from .summary_stats import SummaryStats
from .seeds import set_all_seeds

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from joblib import dump


def get_train_test_split(folder):
    data, labels = get_data(folder)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

    return X_train, X_test, y_train, y_test


def evaluate(X_test, y_test, model):
    y_pred = model.predict(X_test)

    return y_pred, accuracy_score(y_test, y_pred)


def tune(X_train, y_train, param_grid, output):
    estimator = Pipeline([
        ("summary_stats", SummaryStats()),
        ("scaler", MinMaxScaler()),
        ("pca", PCA()),
        ("classifier", KNeighborsClassifier())
    ])

    clf = GridSearchCV(estimator, param_grid, cv=5, verbose=1, n_jobs=-1)
    search = clf.fit(X_train, y_train)

    dump(search, output)


if __name__ == "__main__":
    set_all_seeds(0)
    param_grid = {
        "summary_stats__n_mels" : np.array([40]),
        "pca__n_components" : np.array([32]),
        "classifier__n_neighbors" : np.array([2])
    }
    data, labels = get_data('./recordings')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
    tune(X_train, y_train, param_grid, './searches/grid_search.joblib')