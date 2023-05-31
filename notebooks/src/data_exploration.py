import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from prettytable import PrettyTable
from pandas.api.types import is_integer_dtype, is_float_dtype
from IPython.display import display


def check_float_is_int(s: pd.Series) -> bool:
    is_integer = s.apply(
        lambda x: True if (x.is_integer()) | (np.isnan(x)) else False
    ).all()
    return is_integer


def is_numeric(s: pd.Series, numeric_unq: int = None) -> bool:
    n_unq = s.nunique()
    if numeric_unq is None:
        n_filled = s.notna().sum()
        n_percs = n_unq / n_filled
        if n_percs > 0.01:
            return True
    else:
        if n_unq > numeric_unq:
            return True
    return False


def get_col_type(s: pd.Series, numeric_unq: int = None) -> bool:
    if is_integer_dtype(s):
        if is_numeric(s, numeric_unq):
            return "numeric"
        else:
            return "ordinal"
    elif is_float_dtype(s):
        if check_float_is_int(s):
            if is_numeric(s, numeric_unq):
                return "numeric"
            else:
                return "ordinal"
        else:
            return "numeric"
    else:
        return "nominal"


def report_number_na(df: pd.DataFrame, target: str, indx: str = None):
    """
    Number of missing values in target
    and number of rows with at least one NaN

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    target : str
        [description]
    indx : str, optional
        [description], by default None
    """
    target_na = df[target].isna()
    rows_na = df.isna().any(axis=1).sum()

    report = PrettyTable()
    report.field_names = ["Metric", "Value"]
    report.add_row([f"# NaN in {target}", target_na.sum()])
    report.add_row(["# rows with at least one NaN", rows_na])
    print(report)

    if indx is None:
        x = df.index
        x_name = "index"
    else:
        x = df[indx]
        x_name = indx

    if target_na.sum() > 0:
        plt.plot(x, target_na)
        plt.title(f"Distribution of NaN in {target} VS {x_name}")
        plt.xlabel(x_name)
        plt.ylabel(f"Boolean: 1 = NaN in {target}")
        plt.show()


def print_col_types(col_types: dict, col_type: str = None):
    def print_col_type(col_type, col_set):
        print(col_type)
        print("\t" + "\n\t".join(col_set))

    if col_type is None:
        for col_type, col_set in col_types.items():
            print_col_type(col_type, col_set)
    else:
        col_set = col_types[col_type]
        print_col_type(col_type, col_set)


def report_basic_descr(df, n_show: int = 3, numeric_unq: int = None):
    """
    Reports number of
    - unique numbers
    - number of filled (not missed) values
    - column type
    - sample of smallest values
    - sample of largets values

    Parameters
    ----------
    df : [type]
        [description]
    n_show : int, optional
        [description], by default 3
    """
    # sort by number of unq values
    # and remove all cols where unq values more than max_cats
    #  report = PrettyTable()
    report = pd.DataFrame(
        columns=[
            "Col", "# Unq Vals",
            "# Filled Vals", "type",
            "Head values", "Tail values"
        ]
    ).set_index("Col")
    col_types = {
        "numeric": set(),
        "ordinal": set(),
        "nominal": set()
    }
    cols_const = list()

    for col in df.columns:
        col_type = get_col_type(df[col], numeric_unq)
        col_types[col_type].add(col)
        n_unq = df[col].nunique()
        if n_unq <= 1:
            cols_const.append(col)
            continue
        n_filled = df[col].notna().sum()
        unq = df[col].sort_values().drop_duplicates().values
        n_show_tail = max(min((n_unq - n_show, n_show)), 0)
        unq_head = [str(x) for x in unq[:n_show]]
        unq_head = ", ".join([
            x if (len(x) <= 50) else x[:48] + ".." for x in unq_head
        ])
        unq = df[col].sort_values(ascending=False).drop_duplicates().values
        unq_tail = [str(x) for x in unq[:n_show_tail]]
        unq_tail = ", ".join([
            x if (len(x) <= 50) else x[:48] + ".." for x in unq_tail
        ])
        report.loc[col] = [
            n_unq, n_filled, col_type, unq_head, unq_tail
        ]

    display(report.sort_values(by=["type", "# Unq Vals"]))
    return report, col_types, cols_const


def scatter_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str = None,
    title: str = "Scatter plot",
    mode: str = "markers"
):
    """
    Draws scatter plot x VS y and
    colorized if needed by color_col

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    x : str
        [description]
    y : str
        [description]
    color_col : str, optional
        [description], by default None
    title : str, optional
        [description], by default "Scatter plot"
    """
    fig = go.Figure()
    if color_col is None:
        fig.add_trace(go.Scattergl(
            x=df[x], y=df[y], mode=mode,
            name=f"{x} VS {y}",
        ))
    else:
        fig.add_trace(go.Scattergl(
            x=df[x], y=df[y], mode=mode,
            name=f"{x} VS {y}",
            marker_color=df[color_col],
            text=df[color_col]
        ))
    fig.update_layout(
        title=title,
    )
    fig.show()


def get_nan_by_cols(df: pd.DataFrame, col_types: dict = None) -> pd.DataFrame:
    """
    output contains number of missing values in columns
    and type of column

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    col_types : dict, optional
        [description], by default None

    Returns
    -------
    pd.DataFrame
        [description]
    """
    if col_types is None:
        col_types_reverse = {
            col: df[col].dtype for col in df.columns
        }
    else:
        col_types_reverse = {
            col: col_type
            for col_type, col_list in col_types.items()
            for col in col_list
        }

    df_nan = df.isna().sum().sort_values(ascending=False).rename("# Nan")
    df_nan = df_nan.loc[df_nan > 0]
    df_nan = df_nan.to_frame()
    if col_types is None:
        return df_nan
    else:
        col_type_ser = df_nan.apply(
            lambda x: col_types_reverse[x.name]
            if x.name in col_types_reverse else "Unknown", axis=1
        )
        col_type_ser = col_type_ser.rename("col_type")
        return df_nan.join(col_type_ser)


def plot_cat_heatmap(df: pd.DataFrame, col1: str, col2: str):
    df = df.loc[:, [col1, col2]].copy()
    df = df.groupby([col1, col2]).size()
    df = df.to_frame().reset_index().pivot(index=[col1], columns=[col2])
    sns.heatmap(df)
