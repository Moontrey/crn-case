import numpy as np
import pandas as pd
from scipy.stats.kde import gaussian_kde
from sklearn.decomposition import PCA
from pathlib import Path


SHOW = False


def get_kde(s: pd.Series, size: int = 1000):
    # kde
    if s.shape[0] > size:
        vals = np.random.choice(s.dropna(), size)
    else:
        vals = s.dropna()
        size = s.shape[0]
    range_col = np.linspace(np.min(vals), np.max(vals), size)
    kde_f = gaussian_kde(vals)
    kde_col = kde_f(range_col)
    
    return range_col, kde_col


def get_img_path():
    img_path = Path("./img/")
    img_path.mkdir(exist_ok=True, parents=True)

    return img_path
