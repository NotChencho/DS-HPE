from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any


def get_random_forest(n_estimators=100, max_depth=None, random_state=42, **kwargs):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        **kwargs
    )


def get_gradient_boosting(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, **kwargs):
    base_estimator = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )
    return MultiOutputRegressor(base_estimator)


def get_xgboost(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, **kwargs):
    base_estimator = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        tree_method='hist',
        **kwargs
    )
    return MultiOutputRegressor(base_estimator)


def get_ridge(alpha=1.0, random_state=42, **kwargs):
    return Ridge(alpha=alpha, random_state=random_state, **kwargs)


def get_elastic_net(alpha=1.0, l1_ratio=0.5, random_state=42, **kwargs):
    base_estimator = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        max_iter=2000,
        **kwargs
    )
    return MultiOutputRegressor(base_estimator)


def get_svr(kernel='rbf', C=1.0, epsilon=0.1, **kwargs):
    base_estimator = SVR(kernel=kernel, C=C, epsilon=epsilon, **kwargs)
    return MultiOutputRegressor(base_estimator)