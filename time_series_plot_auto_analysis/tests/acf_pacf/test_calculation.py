import pytest
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample

from time_series_plot_auto_analysis.acf_pacf.calculation \
    import add_syntheric_trend


class TestCalculation:

    def test_trend(self):
        arparams = np.array([0.7, -0.5])
        arparams = np.r_[1, -arparams]
        maparams = np.array([0.3, -0.4])
        maparams = np.r_[1, -maparams]
        number_of_samples = 100
        generated_arma_sample = arma_generate_sample(arparams, maparams,
                                                     number_of_samples)
        generated_arma_sample_df = pd.DataFrame(generated_arma_sample)
        trend_span_ratio = 4
        trend_span = ((generated_arma_sample_df.max()
                      - generated_arma_sample_df.min()).values[0]
                          * trend_span_ratio)

        arma_with_trend_df = add_syntheric_trend(generated_arma_sample_df,
                                        trend_span_ratio = trend_span_ratio)

        assert (generated_arma_sample_df.iloc[0].values[0]
                    == arma_with_trend_df.iloc[0].values[0])
        assert (generated_arma_sample_df.iloc[-1].values[0] + trend_span
                        == arma_with_trend_df.iloc[-1].values[0])
