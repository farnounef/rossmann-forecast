import numpy as np
from projet_home.metrics import rmspe


def test_rmspe_perfect_predictions():
    y_true = [100, 200, 300]
    y_pred = [100, 200, 300]

    assert rmspe(y_true, y_pred) == 0.0


def test_rmspe_ignores_zero_sales():
    y_true = [0, 100, 200]
    y_pred = [50, 110, 180]

    expected = np.sqrt((0.1**2 + 0.1**2) / 2)

    assert np.isclose(rmspe(y_true, y_pred), expected)