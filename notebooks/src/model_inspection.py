import pandas as pd
import numpy as np

import plotly.graph_objects as go

from IPython.display import display
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from prettytable import PrettyTable


def report_gs_results(gs: GridSearchCV, n_show: int = 5):
    """
    Shows best parameters found with grid search
    Shows top n_show rows of gs results
    sorted by mean test score

    Parameters
    ----------
    gs : GridSearchCV
        [description]
    n_show : int, optional
        [description], by default 5
    """
    print("\n", ".." * 50, "\n")
    print("Best score: ")
    print(np.round(gs.best_score_, 3))
    print("Best params: ")
    display(pd.Series(gs.best_params_).to_frame())
    print(".." * 50, "\n")

    print("Grid Search Results: ")
    gs_results = pd.DataFrame(gs.cv_results_)
    gs_results.sort_values(by="mean_test_score", ascending=False, inplace=True)
    cols = [
        "mean_train_score",
        "mean_test_score",
        "std_train_score",
        "std_test_score"
        ]
    cols = cols + [x for x in gs_results.columns if "param_" in x]
    display(gs_results[cols].head(n_show))
    print("\n", ".." * 50, "\n")


def report_regression_coef(model, X: pd.DataFrame, n_show: int = 5):
    """
    displays regression coefficiens in "importance" order
    and makes bar plot

    Parameters
    ----------
    model : [type]
        [description]
    X : pd.DataFrame
        [description]
    n_show : int, optional
        [description], by default 5

    Returns
    -------
    [type]
        [description]
    """
    if model.coef_.shape[0] == X.shape[1]:
        model_coef = pd.Series(model.coef_, index=X.columns)
    elif model.coef_.shape[1] == X.shape[1]:
        model_coef = pd.Series(model.coef_[0], index=X.columns)
    else:
        model_coef = pd.Series(model.coef_, index=X.columns)
    if isinstance(model.intercept_, np.ndarray):
        # some models return array
        model_coef["intercept"] = model.intercept_[0]
    else:
        model_coef["intercept"] = model.intercept_

    order = model_coef.abs().sort_values(ascending=False).index
    order = list(order)
    order.remove("intercept")
    order = ["intercept"] + order
    model_coef = model_coef.loc[order]

    print("\n", ".." * 50, "\n")
    print("Largest Coefficients: ")
    display(model_coef.head(n_show).to_frame())
    print(".." * 50, "\n")
    zero = model_coef == 0
    print(
        "Number of zero coeficients: ", zero.sum(),
        " from ", model_coef.shape[0]
    )
    if zero.sum() > 0:
        print("\nSome zero coefficients: ",)
    zerp_coef_str = "\n * ".join(
        [""] + [str(x) for x in zero.index[zero.values][:n_show]]
    )
    print(zerp_coef_str)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=model_coef.head(n_show).values,
            y=model_coef.head(n_show).index,
            orientation='h',
            name=""
        ))
    fig.update_layout(
        title="Feature importance (from coefficients values)"
    )
    fig.show()

    return model_coef


def plot_perm_imp(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 20,
    n_show: int = 10,
    scoring=None
):
    """
    The permutation feature importance
    is defined to be the decrease
    in a model score when
    a single feature value
    is randomly shuffled
    https://scikit-learn.org/stable/modules/permutation_importance.html

    Parameters
    ----------
    model : [type]
        [description]
    X : pd.DataFrame
        [description]
    y : pd.Series
        [description]
    n_repeats : int, optional
        [description], by default 20
    n_show : int, optional
        [description], by default 10
    scoring : str or scorer
    """
    importance_perm = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring=scoring
    )
    sorted_idx = importance_perm.importances_mean.argsort()
    importance_perm = pd.DataFrame(
        importance_perm.importances[sorted_idx].T,
        index=range(n_repeats), columns=X.columns[sorted_idx]
    )
    importance_perm = importance_perm.iloc[:, : n_show]
    fig = go.Figure()
    for col in importance_perm.columns:
        fig.add_trace(go.Box(x=importance_perm[col], name=col))
    fig.update_layout(
        title="Feature importance (permutation-based)"
    )
    fig.show()


def plot_MDI(model, X: pd.DataFrame, n_show: int = 10):
    """
    impurity-based feature importance 

    display feature importance from tree-based algs

    The relative rank (i.e. depth) of a feature used
    as a decision node in a tree can be used
    to assess the relative importance
    of that feature with respect to the predictability
    of the target variable.
    Features used at the top of the tree contribute
    to the final prediction decision
    of a larger fraction of the input samples

    Parameters
    ----------
    model : [type]
        [description]
    X : pd.DataFrame
        [description]
    n_show : int, optional
        [description], by default 10
    """
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=feature_importance[sorted_idx][:n_show],
            y=X.columns[sorted_idx][:n_show],
            orientation='h',
            name=""
        ))
    fig.update_layout(
        title="Feature importance (impurity-based)"
    )
    fig.show()


def report_results(yhat, yhat_test, y, y_test, scorer_dict: dict, r=2):
    """
    reports metrics for train and test cases
    and returns residuals

    Parameters
    ----------
    yhat : [type]
        [description]
    yhat_test : [type]
        [description]
    y : [type]
        [description]
    y_test : [type]
        [description]
    scorer_dict : dict
        [description]
    r : int, optional
        [description], by default 2

    Returns
    -------
    pd.DataFrame:
        columns: e, y, y_hat, is_test
    """
    report = PrettyTable()
    report.field_names = ["Dataset", "Metric", "Value"]
    for name, scorer in scorer_dict.items():
        report.add_row(["Train", name, np.round(scorer(y, yhat), r)])
        report.add_row(["Test", name, np.round(scorer(y_test, yhat_test), r)])

    df = pd.DataFrame(
        {
            "e": np.concatenate([(y - yhat), (y_test - yhat_test)]),
            "y": np.concatenate([y, y_test]),
            "y_hat": np.concatenate([yhat, yhat_test]),
            "is_test": [0] * len(y) + [1] * len(y_test)
        }
    )
    print(report)

    return df


def add_output_type(
    df: pd.DataFrame,
    y: np.array,
    yhat: np.array,
    new_col: str = "prediction",
    join_correct: bool = False
):
    df.loc[:, new_col] = ""

    if join_correct:
        name_correct = "correct"
    else:
        name_correct = "correct_{}"

    for i in np.unique(y):
        for j in np.unique(yhat):
            mask = (y == i) & (yhat == j)
            if i == j:
                name = name_correct.format(i)
            else:
                name = f"true_{i}_pred_{j}"
              
            df.loc[mask, new_col] = name

    return df
