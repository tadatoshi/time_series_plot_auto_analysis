import pytest
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample

from time_series_plot_auto_analysis.acf_pacf.calculation \
    import add_syntheric_trend, add_synthetic_seasonality


class TestCalculation:

    @pytest.fixture()
    def generated_arma_sample_df(self):
        arparams = np.array([0.7, -0.5])
        arparams = np.r_[1, -arparams]
        maparams = np.array([0.3, -0.4])
        maparams = np.r_[1, -maparams]
        number_of_samples = 100
        generated_arma_sample = arma_generate_sample(arparams, maparams,
                                                     number_of_samples)
        yield pd.DataFrame(generated_arma_sample)

    def test_trend(self, generated_arma_sample_df):
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

    def test_seasonality(self, generated_arma_sample_df):
        regular_period_p = 15 # 15 minutes, assuming time is in unit of minute
        wave_number_n = 3 # n in 2 * pi * n * t or omega in some expression.

        arma_with_seasonality_df = add_synthetic_seasonality(
            generated_arma_sample_df, regular_period_p, wave_number_n)

        # 1 comes from cos max value:
        assert (generated_arma_sample_df.iloc[0].values[0] + 1
                    == arma_with_seasonality_df.iloc[0].values[0])
        assert (pytest.approx(generated_arma_sample_df.iloc[regular_period_p]
                              .values[0] + 1)
                == arma_with_seasonality_df.iloc[regular_period_p]
                   .values[0])
