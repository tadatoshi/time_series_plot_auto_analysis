import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def fit_model_and_get_residual(generated_arma_sample_df: pd.DataFrame,
                               arima_order):
    arma_model = ARIMA(generated_arma_sample_df, order=arima_order)
    arma_model_result = arma_model.fit()
    return arma_model_result.resid.to_frame()
