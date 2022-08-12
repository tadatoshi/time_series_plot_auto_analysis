import pytest
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample

from time_series_plot_auto_analysis.diagnostic.calculation import fit_model_and_get_residual


class TestCalculation:

    @pytest.fixture()
    def generated_arma_sample_info(self):
        arparams = np.array([0.7, -0.5])
        full_arparams = np.r_[1, -arparams]
        maparams = np.array([0.3, -0.4])
        full_maparams = np.r_[1, -maparams]
        number_of_samples = 100
        generated_arma_sample = arma_generate_sample(
            full_arparams, full_maparams, number_of_samples)
        arima_order = (len(arparams), 0, len(maparams))
        return pd.DataFrame(generated_arma_sample), arima_order

    def test_fit_model_and_get_residual(self, generated_arma_sample_info):
        generated_arma_sample_df = generated_arma_sample_info[0]
        arima_order = generated_arma_sample_info[1]

        residuals = fit_model_and_get_residual(generated_arma_sample_df,
                                               arima_order)

        assert type(residuals) is pd.DataFrame
        assert len(residuals) == len(generated_arma_sample_df)
        assert (residuals.iloc[0].values[0] !=
                generated_arma_sample_df.iloc[0].values[0])
