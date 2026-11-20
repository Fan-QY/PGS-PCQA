"""Standard PCQA metrics, aligned with PMP / DSP / SAF-Net: SRCC and KRCC on raw predictions; PLCC and
RMSE after the 5-parameter logistic mapping (VQEG), computed per test fold and averaged over folds."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr, kendalltau


def logistic5(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1 / (1 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def fit5(pred, mos):
    pred, mos = np.asarray(pred, float), np.asarray(mos, float)
    p0 = [np.max(mos), 1.0, np.mean(pred), 0.1, np.mean(mos)]
    try:
        return logistic5(pred, *curve_fit(logistic5, pred, mos, p0=p0, maxfev=50000)[0])
    except RuntimeError:  # fall back to a linear map when the fit does not converge
        return np.polyval(np.polyfit(pred, mos, 1), pred)


def fold_metrics(mos, pred):
    y = fit5(pred, mos)
    return {'SRCC': spearmanr(mos, pred)[0], 'PLCC': pearsonr(mos, y)[0], 'KRCC': kendalltau(mos, pred)[0],
            'RMSE': float(np.sqrt(np.mean((np.asarray(mos) - y) ** 2)))}


def cv_metrics(df):
    """df with columns fold, mos, pred -> mean of per-fold metrics."""
    rows = [fold_metrics(g.mos.values, g.pred.values) for _, g in df.groupby('fold')]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
