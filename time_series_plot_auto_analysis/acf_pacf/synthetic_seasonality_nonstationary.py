from time_series_plot_auto_analysis.acf_pacf.calculation \
    import add_synthetic_seasonality
from time_series_plot_auto_analysis.acf_pacf.synthetic \
    import SyntheticAcfPacfPlot, SyntheticAcfPlot, SyntheticPacfPlot


class SeasonalityNonstationaryAcfPacfPlot(SyntheticAcfPacfPlot):

    def _synthetic_acf_plot(self):
        return SeasonalityNonstationaryAcfPlot()

    def _synthetic_pacf_plot(self):
        return SeasonalityNonstationaryPacfPlot()


class SeasonalityNonstationaryAcfPlot(SyntheticAcfPlot):

    def _directory_file_name_prefix(self):
        return "synthetic_seasonality_nonstationary_acf"

    def _additional_calculation(self, generated_arma_sample_df, lags=15):
        return add_synthetic_seasonality(generated_arma_sample_df)


class SeasonalityNonstationaryPacfPlot(SyntheticPacfPlot):

    def _directory_file_name_prefix(self):
        return "synthetic_seasonality_nonstationary_pacf"

    def _additional_calculation(self, generated_arma_sample_df, lags=15):
        return add_synthetic_seasonality(generated_arma_sample_df)
