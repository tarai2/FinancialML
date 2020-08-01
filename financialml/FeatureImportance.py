import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection._split import KFold
import numba


def RandomForestClassifier():
    # RandomForest
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_features=1,  # 分割に用いるfeature数
        class_weight='balanced',
        min_weight_fraction_leaf=0
    )
    clf = BaggingClassifier(
        base_estimator=clf,
        n_estimators=1000,
        max_features=1.,
        max_samples=1.,
        oob_score=False
    )
    return clf


def calculateMDI(X, y, clf=None):
    """ Mean Decrease Impurity
    Args:
        X (pd.DataFrame): feature
        y (pd.Series): label
        clf (estimator, optional): sklearn's classifier, that have .estimators_, and .feature_importance_.

    Returns:
        pd.DataFrame: each feature importance's mean and std.dev
    """
    if clf is None:
        clf = RandomForestClassifier()

    model = clf.fit(X, y)
    indivisualImportance = pd.DataFrame(
        [tree_i.feature_importances_ for tree_i in model.estimators_],
        columns=X.columns
    ).replace(0, np.nan)  # 全く使われなかったFeatureはnanに

    # 平均,誤差のDataFrameにして合計1に規格化
    importance = pd.concat({
        "mean": indivisualImportance.mean(),
        "std": indivisualImportance.std()*indivisualImportance.shape[0]**(-0.5)}, axis=1
    )
    importance = importance / importance["mean"].sum()
    return importance


def calculateMDA(X, y, nSplit=10, clf=None, scoring="NegAL"):
    """
    Args:
        clf (classifier): model that have fit() and predict_proba()
        X (pd.DataFrame): feature
        y (pd.Series): label
        nSplit (int, optional): # of cv splits.

    Returns:
        pd.DataFrame: decrease of scoring metric.
    """
    assert scoring in ["NegAL", "negative_log_loss"],\
        f"scoring function '{scoring}' is not supported"
    if clf is None:
        clf = RandomForestClassifier()

    cvGen = KFold(n_splits=nSplit)
    y_dummied = pd.get_dummies(y)

    # ISスコア, 各Featureをshuffle時のOOSスコア を計算
    score_in, score_out = pd.Series(dtype=float), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0 = X.iloc[train, :], y.iloc[train]
        X1, y1 = X.iloc[test, :], y.iloc[test]
        y1_dummied = y_dummied.iloc[test].values
        fit = clf.fit(X=X0, y=y0)
        predict_prob = fit.predict_proba(X1)
        score_in.loc[i] = -log_loss(y1, predict_prob, labels=clf.classes_)
        if scoring == "NegAL":
            score_in.loc[i] = (y1_dummied * predict_prob).mean()
        elif scoring == "negative_log_loss":
            score_in.loc[i] = -log_loss(y1_dummied.values, predict_prob)
        # 各々のFeatureをランダムシャッフルしてOOSスコアを計算
        for colname in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[colname].values)
            predict_prob = fit.predict_proba(X1_)
            if scoring == "NegAL":
                score_out.loc[i, colname] = (y1_dummied * predict_prob).mean()
            elif scoring == "negative_log_loss":
                score_out.loc[i, colname] = -log_loss(y1_dummied.values, predict_prob)

    # 各cvでのscoreの減少を計算
    importance = (-score_out).add(score_in, axis=0)
    importance = pd.concat({
        "mean": importance.mean(),
        "std": importance.std()*importance.shape[0]**(-0.5)}, axis=1
    )
    return importance
