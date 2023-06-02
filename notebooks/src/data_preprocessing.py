import pandas as pd
import numpy as np

from pandas import CategoricalDtype

from pydantic import BaseModel
from sklearn.exceptions import NotFittedError
from sklearn.base import TransformerMixin

from typing import Tuple


def remove_columns_inplace(df: pd.DataFrame, columns, col_types):
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        for col_type in col_types.keys():
            if col in col_types[col_type]:
                col_types[col_type].remove(col)
                break

    if df is None:
        return
    else:
        columns_to_drop = set(columns) & set(df.columns)
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)


def move_columns(columns, new_type, col_types):
    '''
    Move colum from one type to another
    '''
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        for col_type in col_types.keys():
            if col in col_types[col_type]:
                col_types[col_type].remove(col)
                break

        col_types[new_type].add(col)


def add_columns_types(columns, new_type, col_types):
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        col_types[new_type].add(col)


def str_to_int(s: pd.Series, replace: dict) -> pd.Series:
    s_new = s.apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

    for what_s, for_s in replace.items():
        s_new = s_new.apply(
            lambda x: x.replace(what_s, for_s) if isinstance(x, str) else x
        )

    s_new = s_new.astype(float)

    return s_new


def make_ordered_categorical(
    df: pd.DataFrame, col: str, order: list = None
):
    """
    replaces columns col
    with pd.Categorical ordered

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    col : str
        [description]
    order : list, optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    df[col] = pd.Series(
        pd.Categorical(df[col], categories=order, ordered=True),
        index=df.index
    )
    return df


def make_categorical(df: pd.DataFrame, col: str):
    """
    replaces columns col
    with pd.Categorical

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    col : str
        [description]

    Returns
    -------
    [type]
        [description]
    """
    df[col] = pd.Categorical(df[col], ordered=False)
    return df


def get_code_from_ordinal(
    df: pd.DataFrame, cols: list
) -> pd.DataFrame:
    """
    Replaces values of categories with their code

    Returns
    -------
    [type]
        [description]
    """
    if isinstance(cols, str):
        cols = [cols]
    df = df.copy()
    cols = set(cols) & set(df.columns)

    for col in cols:
        if isinstance(df[col].dtype, CategoricalDtype):
            df[col] = df[col].cat.codes
        else:
            print(f"Cannot convert {col} of type {df[col].dtype}")
    return df


def add_binary_nan_col(
    df: pd.DataFrame, col: str
) -> Tuple[pd.DataFrame, str]:
    """
    Adds binary column
    which shows where column contains NaN values

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    col : str
        [description]

    Returns
    -------
    pd.DataFrame, str
        new dataframe and column name
    """
    df = df.copy()
    if df[col].isna().sum() > 0:
        new_col = f"{col}_nan"
        df[new_col] = 1 * df[col].isna()
        print(f"New column {new_col} is added")
    else:
        print(df[col].isna().sum())
        new_col = None
    return df, new_col


def get_columns_with_high_corr(X: pd.DataFrame, max_corr: float = 1):
    """
    Gets one of two columns
    which have high correlations

    Parameters
    ----------
    X : pd.DataFrame
        [description]
    max_corr : float, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    """
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        column for column in upper_tri.columns
        if any(upper_tri[column] >= max_corr)
    ]
    return to_drop


def get_joined_cats(s: pd.Series, n: int = 10) -> pd.Series:
    s_vc = s.value_counts()
    cats_join = s_vc.loc[s_vc < n].index
    s_new = s.replace({
        x: "other_cat" for x in cats_join
    })
    return s_new


def join_cats_by_name(df, col, cats, new_name):
    df = df.copy()
    replace_dict = {
        x: new_name for x in cats
    }
    df[col] = df[col].replace(replace_dict)
    
    return df


class FillNanConfig(BaseModel):
    drop_col: list = []
    drop_row: list = []
    mode: list = []
    median: list = []
    mean: list = []
    max_val: list = []
    min_val: list = []
    fill: dict = dict()
    fill_frm_col: dict = dict()


class FillNan(TransformerMixin):
    """
    Fills missing values

    Parameters
    ----------
    TransformerMixin : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotFittedError
        [description]
    """
    params: FillNanConfig
    fitted: bool

    def __init__(self, params: FillNanConfig or dict):
        if isinstance(params, dict):
            params = FillNanConfig(**params)
        self.params = params
        self.fitted = False

    def _map_steps(self):
        return {
            "drop_col": self._drop_col,
            "drop_row": self._drop_row,
            "mode": self._fill_mode,
            "median": self._fill_median,
            "mean": self._fill_mean,
            "max_val": self._fill_max_val,
            "min_val": self._fill_min_val,
            "fill": self._fill,
            "fill_frm_col": self._fill_frm_col
        }

    def _drop_col(self, df, step_params: dict = None):
        if self.fitted:
            cols = set(self.params.drop_col) & set(df.columns)
            df.drop(columns=cols, inplace=True)

        return step_params

    def _drop_row(self, df, step_params: dict = None):
        if self.fitted:
            indx = set(self.params.drop_row) & set(df.index)
            df.drop(index=indx, inplace=True)

        return step_params

    def _fill_mode(self, df, step_params: dict = None):
        cols = set(self.params.mode) & set(df.columns)

        if step_params is None:
            step_params = df[cols].mode().iloc[0].to_dict()

        if self.fitted:
            df.fillna(step_params, inplace=True)
        return step_params

    def _fill_median(self, df, step_params: dict = None):
        cols = set(self.params.median) & set(df.columns)

        if step_params is None:
            step_params = df[cols].median().to_dict()

        if self.fitted:
            df.fillna(step_params, inplace=True)
        return step_params

    def _fill_mean(self, df, step_params: dict = None):
        cols = set(self.params.mean) & set(df.columns)

        if step_params is None:
            step_params = df[cols].mean().to_dict()

        if self.fitted:
            df.fillna(step_params, inplace=True)
        return step_params

    def _fill_max_val(self, df, step_params: dict = None):
        cols = set(self.params.max_val) & set(df.columns)

        if step_params is None:
            step_params = df[cols].max().to_dict()

        if self.fitted:
            df.fillna(step_params, inplace=True)
        return step_params

    def _fill_min_val(self, df, step_params: dict = None):
        cols = set(self.params.min_val) & set(df.columns)

        if step_params is None:
            step_params = df[cols].min().to_dict()

        if self.fitted:
            df.fillna(step_params, inplace=True)
        return step_params

    def _fill(self, df, step_params: dict = None):
        if self.fitted:
            fill_filt = {
                k: v for k, v in self.params.fill.items()
                if k in df.columns
            }
            df.fillna(fill_filt, inplace=True)

        return step_params

    def _fill_frm_col(self, df, step_params: dict = None):
        if self.fitted:
            fill_filt = {
                k: v for k, v in self.params.fill_frm_col.items()
                if (k in df.columns) & (v in df.columns)
            }

            for col_to_fill, col_from in fill_filt.items():
                mask = (
                    df.loc[:, col_to_fill].isna()
                    & df.loc[:, col_from].notna()
                )
                df.loc[mask, col_to_fill] = df.loc[mask, col_from]

        return step_params

    def fit(self, X, y=None):
        df = X.copy()
        params_fitted = dict()

        for step_name, step_fun in self._map_steps().items():
            step_params = step_fun(df)
            params_fitted[step_name] = step_params

        self.fitted = True
        self.params_fitted = params_fitted
        return self

    def transform(self, X, copy=None):
        if self.fitted is False:
            raise NotFittedError
        if copy:
            df = X.copy()
        else:
            df = X

        params_fitted = self.params_fitted
        for step_name, step_fun in self._map_steps().items():
            _ = step_fun(df, params_fitted[step_name])

        return df


class JoinCatsConfig(BaseModel):
    thresh: float or int = 0.1
    columns: list


class JoinCats(TransformerMixin):
    """
    checks if several categories of column
    have less than params.thresh % of data
    if yes then merges such categories

    Parameters
    ----------
    TransformerMixin : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotFittedError
        [description]
    """
    params: JoinCatsConfig
    fitted: bool

    def __init__(self, params: JoinCatsConfig or dict = None):
        if isinstance(params, dict):
            params = JoinCatsConfig(**params)
        self.params = params
        self.fitted = False

    def fit(self, X, y=None):
        if self.params.thresh < 1:
            n = int(X.shape[0] * self.params.thresh)
        else:
            n = int(self.params.thresh)

        cats_to_join = dict()
        cols = set(X.columns) & set(self.params.columns)

        for col in cols:
            val_cnt = X[col].value_counts()

            if (val_cnt < n).any():
                cats = val_cnt.index[val_cnt < n]
                if len(cats) > 1:
                    cats_to_join[col] = {
                        cat: "other_cat" for cat in cats
                    }

        self.fitted = True
        self.cats_to_join = cats_to_join
        return self

    def transform(self, X, copy=None):
        if self.fitted is False:
            raise NotFittedError
        if copy:
            X = X.copy()
        else:
            X = X

        cats_to_join = self.cats_to_join
        X = X.replace(cats_to_join)

        return X

    def print_joins(self):
        if self.fitted:
            for col, dict_join in self.cats_to_join.items():
                print("." * 50)
                print(col)
                print("\t" + "\n\t".join([str(x) for x in dict_join.keys()]))


class DropImbalanceColsConfig(BaseModel):
    thresh: float or int = 0.9
    columns: list = []


class DropImbalanceCols(TransformerMixin):
    """
    Drops columns
    where one of categories correspond to
    more than params.thresh % of data

    Parameters
    ----------
    TransformerMixin : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotFittedError
        [description]
    """
    params: DropImbalanceColsConfig
    fitted: bool

    def __init__(self, params: DropImbalanceColsConfig or dict):
        if isinstance(params, dict):
            params = DropImbalanceColsConfig(**params)
        self.params = params
        self.fitted = False

    def fit(self, X, y=None):
        if self.params.thresh < 1:
            n = int(X.shape[0] * self.params.thresh)
        else:
            n = int(self.params.thresh)

        cols_to_drop = list()
        cols = set(X.columns) & set(self.params.columns)

        for col in cols:
            max_cnt = X[col].value_counts().iloc[0]

            if max_cnt > n:
                cols_to_drop.append(col)

        self.fitted = True
        self.cols_to_drop = cols_to_drop
        return self

    def transform(self, X, copy=None):
        if self.fitted is False:
            raise NotFittedError
        if copy:
            X = X.copy()
        else:
            X = X

        cols_to_drop = set(self.cols_to_drop) & set(X.columns)

        return X.drop(columns=cols_to_drop)


def get_log_col(df, col):
    df = df.copy()
    new_col = f"{col}_log"
    if df[col].min() <= 0:
        shift = -df[col].min() + 0.001
        df[new_col] = np.log(df[col] + shift)
    else:
        df[new_col] = np.log(df[col])

    return df, new_col


def get_quantile_bins(df, col, n_bins=10, eps=1):
    df = df.copy()
    new_col = f"{col}_bins"
    bins = df[col].quantile(np.linspace(0, 1, n_bins))
    bins = sorted(list(set(bins)))
    bins[0] = bins[0] - eps
    bins[-1] = bins[-1] + eps
    df[new_col] = pd.cut(df[col], bins)

    return df, new_col


def describe_distances(distance):
    if distance <= 3:
        return 'close'
    
    elif (distance > 3) & (distance <= 10):
        return 'medium'
    
    elif distance > 10:
        return 'far'

