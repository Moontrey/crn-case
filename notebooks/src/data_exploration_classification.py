import pandas as pd
import os
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from scipy.stats import mode
from src.utils import get_kde, get_img_path
from typing import Union
from IPython.display import display


def plot_one_category(
    df_orig: pd.DataFrame,
    target: Union[str, pd.Series],
    col: list or str = "all",
    max_n: int = 10,
    max_cats: int = 20,
    r: int = 3,
    bar_size: int = 15,
    norm: str = "target",
    html_file=None
):
    df = df_orig.copy()

    if isinstance(target, pd.Series):
        df[target.name] = target
        target = target.name

    df[target] = df[target].astype("str")

    df.fillna("NaN", inplace=True)
    unq_target = df[target].unique()
    if col == target:
        cardinality = len(unq_target)
        val_cnt = df[col].value_counts()
        fig = make_subplots(rows=1, cols=2)
        fig.append_trace(
            go.Bar(
                y=val_cnt.index,
                x=val_cnt.values,
                orientation='h',
                name=col
            ),
            row=1, col=1
        )

        fig.update_layout(
            title=f"(1) Distributon of {col}",
            width=900,
            height=200 + bar_size * cardinality,
            margin=dict(
                l=50,
                r=50,
                b=50,
                t=50,
                pad=10
            ),
            yaxis_title="%",
            title_font_size=10
        )
        if html_file is None:
            fig.show()
        else:
            html_file.write(
                fig.to_html(full_html=False, include_plotlyjs='cdn')
            )

    else:
        val_cnt = df[col].value_counts(normalize=True)
        if val_cnt.shape[0] > max_n:
            replace_dict = {
                x: "Other cats" for x in val_cnt.index[max_n:]
            }
            df[col] = df[col].replace(replace_dict)

        order = val_cnt.index
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.25)
        val_cnt = df[col].value_counts()
        fig.append_trace(
            go.Bar(
                y=val_cnt.index,
                x=val_cnt.values,
                orientation='h',
                name=col
            ),
            row=1, col=1
        )

        val_cnt = df[[target, col]].groupby([target, col]).size()
        val_cnt_indx = [
            (l1, l2) for l2 in order
            for (l1, l2_) in val_cnt.index if l2 == l2_
            ]
        val_cnt = val_cnt.loc[val_cnt_indx]
        if norm == "target":
            val_cnt_norm = (
                val_cnt / val_cnt.groupby(level=1).sum() * 100
            ).round(r)
        elif norm == "col":
            val_cnt_norm = (
                val_cnt / val_cnt.groupby(level=0).sum() * 100
            ).round(r)
        else:
            val_cnt_norm = val_cnt
        cardinality = val_cnt_norm.shape[0]

        for target_cat in unq_target:
            fig.append_trace(
                go.Bar(
                    y=val_cnt_norm.loc[target_cat].index,
                    x=val_cnt_norm.loc[target_cat].values,
                    orientation='h',
                    name=target_cat,
                    text=val_cnt.loc[target_cat].values
                ),
                row=1, col=2
            )

        fig.update_layout(
            title=f"(1) Distributon of {col} by categories."
            + f"\n(2) % of each {col} category with respect "
            + f"to each {target} category",
            width=900,
            height=200 + bar_size * cardinality,
            margin=dict(
                l=70,
                r=70,
                b=50,
                t=50,
                pad=5
            ),
            yaxis_title="%",
            title_font_size=10
        )
        if html_file is None:
            fig.show()
        else:
            html_file.write(
                fig.to_html(full_html=False, include_plotlyjs='cdn')
            )


def plot_cat_values(
    df_orig: pd.DataFrame,
    target: Union[str, pd.Series],
    cols: list or str = "all",
    max_n: int = 10,
    max_cats: int = 20,
    r: int = 3,
    bar_size: int = 15,
    norm: str = "target",
    html_name='categorical_plots.html'
):
    """
    plots for categorical data (barplots)
    and boxplots for target VS categories

    Parameters
    ----------
    df_orig : pd.DataFrame
        [description]
    target : str
        [description]
    cols : listorstr, optional
        [description], by default "all"
    max_n : int, optional
        if more than max_n cats, then join some of them
    max_cats : int, optional
        if more than max_cats, then plot won't be shown
    r : int, optional
        round of percents to digits
    """
    df = df_orig.copy()

    if isinstance(target, pd.Series):
        df[target.name] = target
        target = target.name

    df.fillna("NaN", inplace=True)

    if cols == 'all':
        cols = df.columns
    else:
        cols = cols.copy()

    # sort by number of unq values
    # and remove all cols where unq values more than max_cats
    n_unq_list = list()
    cols_filtered = list()
    cols_high_card = list()
    for col in cols:
        n_unq = df[col].nunique()
        if n_unq <= max_cats:
            n_unq_list.append(n_unq)
            cols_filtered.append(col)
        else:
            print(
                f"Cannot show standart report for {col} "
                + f"as there are {n_unq} categories"
            )
            cols_high_card.append(col)

    # sort by unq number
    cols_sorted = [col for _, col in sorted(zip(n_unq_list, cols_filtered))]
    # plot
    img_path = get_img_path() / html_name
    if img_path.exists():
        os.remove(img_path)
    with open(img_path, 'a') as html_file:
        for col in cols_sorted:
            print(col)
            plot_one_category(
                df, target, col, max_n, max_cats, r, bar_size, norm,
                html_file=html_file
            )

    return cols_high_card


def plot_hist_boxplot(
    fig, row, raw_col, target_col: pd.Series, size, draw_kde: bool
):
    # kde
    if draw_kde:
        range_col, kde_col = get_kde(raw_col, size)

    fig.append_trace(go.Histogram(
        x=raw_col,
        histnorm='probability density',
        name="Distribution"
    ), row=row, col=1)

    if draw_kde:
        fig.append_trace(
            go.Scattergl(x=range_col, y=kde_col, name="KDE distribution"),
            row=row, col=1)

    for cat in target_col.unique():
        mask = target_col == cat
        fig.add_trace(
            go.Box(
                    y=raw_col.loc[mask].values,
                    name=cat,
            ), row=row, col=2
        )


def plot_one_numeric_distr(
    df: pd.DataFrame,
    col: Union[str, pd.Series],
    target: str,
    draw_mode_plot: bool = True,
    size: int = 1000,
    draw_kde_n: int = 50,
    html_file=None
):
    df = df.copy()
    layout_params = dict(
            title=col,
            width=800,
            height=300,
            margin=dict(
                l=50,
                r=50,
                b=50,
                t=50,
                pad=10
            ),
            title_font_size=10
        )

    if isinstance(target, pd.Series):
        df[target.name] = target
        target = target.name

    df[target] = df[target].astype("str")

    print()
    print(col)
    found_modes = dict()

    if draw_mode_plot:
        col_mode = mode(df[col]).mode[0]
        mask = df.loc[:, col] == col_mode
        val_cnt = df[col].value_counts()
        norm_freq = val_cnt.quantile(0.9) * 10

        if mask.sum() > norm_freq:
            fig = make_subplots(rows=2, cols=2)
        else:
            draw_mode_plot = False
            fig = make_subplots(rows=1, cols=2)
    else:
        fig = make_subplots(rows=1, cols=2)

    target_col = df[target]
    draw_kde = True if df[col].nunique() > draw_kde_n else False
    plot_hist_boxplot(fig, 1, df[col], target_col, size, draw_kde)

    if draw_mode_plot:
        # remove mode
        modes = mode(df[col]).mode
        mask = df[col].notna()
        found_modes[col] = modes
        for col_mode in modes:
            print(f"Removed mode {col_mode} for {col}")
            mask = mask & (df.loc[:, col] != col_mode)
        print()

        raw_col_no_mode = df.loc[mask, col]
        plot_hist_boxplot(fig, 2, raw_col_no_mode, target_col, size, draw_kde)

    fig.update_layout(
        **layout_params
    )
    if html_file is None:
        fig.show()
    else:
        html_file.write(
            fig.to_html(full_html=False, include_plotlyjs='cdn')
        )

    if draw_mode_plot:
        print(pd.concat([
            df[col].describe(),
            raw_col_no_mode.rename(
                f"{raw_col_no_mode.name}_no_mode"
            ).describe()
        ], axis=1))
    else:
        print(df[col].describe())
    return found_modes
    

def plot_numeric_distr(
    df: pd.DataFrame,
    cols: list,
    target: str,
    draw_mode_plot: bool = True,
    size: int = 1000,
    draw_kde_n: int = 50,
    html_name='numerical_plots.html'
):
    if isinstance(cols, str):
        cols = [cols]
    found_modes = dict()

    img_path = get_img_path() / html_name
    if img_path.exists():
        os.remove(img_path)

    with open(img_path, 'a') as html_file:
        for col in cols:
            found_modes.update(plot_one_numeric_distr(
                df, col, target, draw_mode_plot, size, draw_kde_n,
                html_file=html_file
            ))
    return found_modes


def plot_hist_hist(
    fig, row, raw_col, target_col: pd.Series, size, draw_kde: bool
):
    # kde
    if draw_kde:
        range_col, kde_col = get_kde(raw_col, size)

    fig.append_trace(go.Histogram(
        x=raw_col,
        histnorm='probability density',
        name="Distribution"
    ), row=row, col=1)

    if draw_kde:
        fig.append_trace(
            go.Scattergl(x=range_col, y=kde_col, name="KDE distribution"),
            row=row, col=1)

    for cat in target_col.unique():
        mask = target_col == cat
        fig.append_trace(go.Histogram(
            x=raw_col.loc[mask],
            histnorm='probability density',
            name=f"Distribution for {cat}"
        ), row=row, col=2)


def plot_one_ordinal_distr(
    df: pd.DataFrame,
    col: str,
    target: Union[str, pd.Series],
    draw_mode_plot: bool = True,
    size: int = 1000,
    draw_kde_n: int = 50,
    html_file=None
):
    layout_params = dict(
        title=col,
        width=800,
        height=300,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=10
        ),
        title_font_size=10
    )
    df = df.copy()

    if isinstance(target, pd.Series):
        df[target.name] = target
        target = target.name

    print()
    print(col)
    found_modes = dict()

    if draw_mode_plot:
        col_mode = mode(df[col]).mode[0]
        mask = df.loc[:, col] == col_mode
        val_cnt = df[col].value_counts()
        norm_freq = val_cnt.quantile(0.9) * 10

        if mask.sum() > norm_freq:
            fig = make_subplots(rows=2, cols=2)
        else:
            draw_mode_plot = False
            fig = make_subplots(rows=1, cols=2)
    else:
        fig = make_subplots(rows=1, cols=2)

    target_col = df[target]
    draw_kde = True if df[col].nunique() > draw_kde_n else False
    plot_hist_hist(fig, 1, df[col], target_col, size, draw_kde)

    if draw_mode_plot:
        # remove mode
        modes = mode(df[col]).mode
        mask = df[col].notna()
        found_modes[col] = modes
        for col_mode in modes:
            print(f"Removed mode {col_mode} for {col}")
            mask = mask & (df.loc[:, col] != col_mode)
        print()

        raw_col_no_mode = df.loc[mask, col]
        plot_hist_hist(fig, 2, raw_col_no_mode, target_col, size, draw_kde)

    fig.update_layout(
        barmode='overlay', **layout_params
    )
    fig.update_traces(opacity=0.75)

    if html_file is None:
        fig.show()
    else:
        html_file.write(
            fig.to_html(full_html=False, include_plotlyjs='cdn')
        )

    if draw_mode_plot:
        print(pd.concat([
            df[col].describe(),
            raw_col_no_mode.rename(
                f"{raw_col_no_mode.name}_no_mode"
            ).describe()
        ], axis=1))
    else:
        print(df[col].describe())

    return found_modes


def plot_ordinal_distr(
    df: pd.DataFrame,
    cols: list,
    target: Union[str, pd.Series],
    draw_mode_plot: bool = True,
    size: int = 1000,
    draw_kde_n: int = 50,
    html_name='ordinal_plots.html'
):
    found_modes = dict()
    if isinstance(cols, str):
        cols = [cols]
    df = df.copy()

    img_path = get_img_path() / html_name

    if img_path.exists():
        os.remove(img_path)

    with open(img_path, 'a') as html_file:
        for col in cols:
            found_modes.update(plot_one_ordinal_distr(
                df, col, target, draw_mode_plot, size, draw_kde_n,
                html_file=html_file
            ))
    return found_modes


def report_cat_columns(
    df_orig, cols, target: Union[str, pd.Series], t=0.05
):
    cats_with_issues = dict()
    df = df_orig.copy()

    if isinstance(target, pd.Series):
        df[target.name] = target
        target = target.name

    df.fillna("NaN", inplace=True)
    if cols == 'all':
        cols = df.columns
    else:
        cols = cols.copy()

    if t < 1:
        las_col_name1 = f"#Cat card-ty < {t*100}%"
        las_col_name2 = f"#Cat card-ty < {t*100}% by trgt"
    else:
        las_col_name1 = f"#Cat card-ty < {t}"
        las_col_name2 = f"#Cat card-ty < {t} by trgt"

    report = pd.DataFrame(columns=[
        "Column", "# of cats", "Largest cat", "Smallest cat",
        las_col_name1, las_col_name2
        ]).set_index("Column")

    # sort by number of unq values
    # and remove all cols where unq
    # values more than max_cats
    n_unq_list = list()
    cols_filtered = list()
    for col in cols:
        n_unq = df[col].nunique()
        if t < 1:
            cnt_vls = df[col].value_counts(normalize=True)
        else:
            cnt_vls = df[col].value_counts()
        if (cnt_vls < t).any():
            n_unq_list.append(n_unq)
            cols_filtered.append(col)

    # sort by unq number
    cols_sorted = [col for _, col in sorted(zip(n_unq_list, cols_filtered))]
    # plot
    for col in cols_sorted:
        if t < 1:
            cnt_vls = df[col].value_counts(normalize=True)
            cnt_vls_trgt = (
                df[[target, col]].groupby([target, col]).size() / df.shape[0]
            )
        else:
            cnt_vls = df[col].value_counts()
            cnt_vls_trgt = df[[target, col]].groupby([target, col]).size()

        small_cats = (cnt_vls < t)
        small_cats_by_target = (cnt_vls_trgt < t).groupby(level=1).any()
        report.loc[col] = ([
            cnt_vls.shape[0],
            cnt_vls.index[0] if len(cnt_vls.index[0]) <= 50
            else cnt_vls.index[0][:48] + "..",
            cnt_vls.index[-1] if len(cnt_vls.index[-1]) <= 50
            else cnt_vls.index[-1][:48] + "..",
            small_cats.sum(), small_cats_by_target.sum()
        ])
        cats_with_issues[col] = {
            "simple": list(small_cats.index[small_cats]),
            "by_target": list(small_cats_by_target.index[small_cats_by_target])
        }
    display(report)
    return cats_with_issues, report
