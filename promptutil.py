""" Utility functions to formulate and execute promts for tabular data. Also other function useful for evaluation.
"""

import numpy as np

#################################################################################################
# from numpy arrays to textual promts
#################################################################################################


def format_data_point(x, feature_names, optional_features=None, add_if_then=False):
    """X1 = 0.35, X2 = 0.82, X3 = 0.33, X4 = -1.30"""
    num_features = None
    if isinstance(
        x, np.ndarray
    ):  # support numpy arrays and lists (or does len work for the array, too?)
        x = np.atleast_1d(x.squeeze())
        num_features = x.shape[0]
    else:
        num_features = len(x)
    prompt = ""
    for idx in range(num_features):
        if len(feature_names) == 0:  # feature value
            prompt = prompt + f"{x[idx]}, "
        else:  # feature name = feature value, feature name = feature value, ...
            if (
                optional_features is not None
                and feature_names[idx] in optional_features
            ):  # optional features are only included if they are not zero
                if x[idx] == 0:
                    continue
            prompt = prompt + f"{feature_names[idx]} = {x[idx]}, "
    prompt = prompt[:-2]  # no comma at the end
    if add_if_then:
        prompt = "IF " + prompt + ", THEN"
    return prompt



#################################################################################################
# utility functions to parse the model output
#################################################################################################


def read_prefix_float(str, default=None):
    """Read a float from a maximum prefix of the string"""
    for i in reversed(range(len(str) + 1)):
        try:
            return float(str[:i].strip())
        except:
            pass
    if default is not None:
        print(f"String {str} does not have a prefix that is a float.")
        return default
    raise ValueError(f"String '{str}' does not have a prefix that is a float.")


def read_postfix_float(str, default=None):
    """Read a float from a maximum postifx of the string"""
    for i in range(len(str)):
        try:
            return float(str[i:].strip())
        except:
            pass
    if default is not None:
        print(f"String {str} does not have a postfix that is a float.")
        return default
    raise ValueError(f"String '{str}' does not have a postfix that is a float.")


def read_float(str, default=None):
    """read any float from the string. will work well if there is exactly a single float in the string."""
    for i in range(len(str)):
        for j in reversed(range(len(str) + 1)):
            try:
                return float(str[i:j].strip())
            except:
                pass
    if default is not None:
        print(f"String {str} does not contain a float.")
        return default
    raise ValueError(f"String '{str}' does not contain a float.")


#################################################################################################
# statistical utility functions
#################################################################################################
from scipy.stats import bootstrap
from sklearn import metrics


def accuracy_95(labels, predictions):
    """Compute the accuracy. Also compute a 95%-confidence interval using a bootstrap method."""
    acc = metrics.accuracy_score(labels, predictions)
    res = bootstrap(
        (np.array(labels), np.array(predictions)),
        metrics.accuracy_score,
        vectorized=False,
        paired=True,
    )
    print(
        f"Accuracy: {acc:.2f}, 95%-Confidence Interval: ({res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f})"
    )
    return acc, res.confidence_interval


def roc_auc(labels, predictions):
    """Compute the AUC score. Also compute a 95%-confidence interval using a bootstrap method."""
    auc = metrics.roc_auc_score(labels, predictions)
    res = bootstrap(
        (np.array(labels), np.array(predictions)),
        metrics.roc_auc_score,
        vectorized=False,
        paired=True,
    )
    print(
        f"AUC: {auc:.2f}, 95%-Confidence Interval: ({res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f})"
    )
    return auc, res.confidence_interval


#################################################################################################
# fit and evaluate models
#################################################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from scipy.stats import uniform


def fit_logistic_regression_cv(
    X_train, y_train, num_splits=7, scoring="roc_auc", random_state=None
):
    """Fit a logistic regression model using cross validation.

    - Scales the input data
    - Uses stratified cross validation (num_splits parameter, default 7)
    - uses a L2 penalty term and randomized search to find the best hyperparameter
    - uses AUC as the scoring metric
    """
    assert (
        y_train.astype(int) == y_train
    ).all(), f"labels must be integers. found {y_train}"
    y_train = y_train.astype(int)
    pipe = Pipeline([("scaler", StandardScaler()), ("logistic", LogisticRegression())])
    param_space = {
        "logistic__C": uniform(0.01, 10),
    }
    clf = RandomizedSearchCV(
        pipe,
        param_space,
        scoring=scoring,
        cv=StratifiedKFold(
            n_splits=num_splits, shuffle=True, random_state=random_state
        ),
        random_state=random_state,
        verbose=0,
    )
    # execute search
    clf.fit(X_train, y_train)
    # print(
    #    f"Fitting Logistic Regression with 7-fold cross validation. AUC: {clf.best_score_:.2f}. Best Hyperparameter: {clf.best_params_['logistic__C']:.2f}"
    # )
    return clf.best_estimator_


def loo_eval(
    X, y, fit_predict, shuffle=True, few_shot=-1, stratified=True, max_points=1000
):
    """Evaluate a model using leave-one-out cross validation.
    This means that we fit a model on all but one data point and then evaluate it on the left out data point.
    This procedure is repeated for up to max_points data points.

    X: data points
    y: labels or regression target
    fit_predict(X_train, y_train, X_test): function that fits a model on X_train, y_train and returns the predictions on X_test.
    Of course this can also mean sending a prompt to a language model, possibly ignoring the training data.
    shuffle: whether to shuffle the data in each step (Default: True)
    few_shot: whether to reduce the training data to few_shot examples.(Default: -1, that is use all data points)
    stratified: whether to use stratified sampling in the few_shot setting. (Default: True)
    max_points: the maximum number of test points. useful for very large data sets (Default: 1000)

    Returns: The predictions for all data points. The user can then compute a metric between y and this result. The ordering of the returned predictions is the same as the ordering of the data points in X.
    """
    if few_shot > 0:
        assert shuffle == True, "few_shot only makes sense if shuffle is True"
    predictions = np.zeros_like(y)
    for idx in range(min(X.shape[0], max_points)):
        X_test = X[idx : idx + 1, :]
        X_train = np.delete(X, idx, axis=0)
        y_train = np.delete(y, idx, axis=0)
        if shuffle:
            permutation = np.random.permutation(X_train.shape[0])
            X_train = X_train[permutation, :]
            y_train = y_train[permutation]
        if (
            few_shot > 0
        ):  # optionally reduce the training data to a given number of few shot examples
            stratify = (
                y_train if stratified else None
            )  # optionally stratify the few shot examples
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=few_shot, stratify=stratify
            )
        predictions[idx] = fit_predict(X_train, y_train, X_test)
    return predictions


#################################################################################################
# misc
#################################################################################################

import json


def pretty_print_messages(messages):
    """Prints openai chat messages in a nice format"""
    for message in messages:
        print(json.dumps(message))
