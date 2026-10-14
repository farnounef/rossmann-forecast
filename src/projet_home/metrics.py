import numpy as np


def rmspe(y_true, y_pred):
    """Erreur quadratique moyenne en pourcentage, calculée sur les ventes réelles > 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))