from time_series_plot_auto_analysis.acf_pacf.calculation import add_syntheric_trend
from time_series_plot_auto_analysis.acf_pacf.synthetic \
    import SyntheticAcfPacfPlot, SyntheticAcfPlot, SyntheticPacfPlot


class TrendNonstationaryAcfPacfPlot(SyntheticAcfPacfPlot):

    def _syntheric_acf_plot(self):
        return TrendNonstationaryAcfPlot()

    def _syntheric_pacf_plot(self):
        return TrendNonstationaryPacfPlot()

class TrendNonstationaryAcfPlot(SyntheticAcfPlot):

    def _directory_file_name_prefix(self):
        return "synthetic_trend_nonstationary_acf"

    def _additional_calculation(self, generated_arma_sample_df, lags=15):
        return add_syntheric_trend(generated_arma_sample_df)

class TrendNonstationaryPacfPlot(SyntheticPacfPlot):

    def _directory_file_name_prefix(self):
        return "synthetic_trend_nonstationary_pacf"

    def _additional_calculation(self, generated_arma_sample_df, lags=15):
        return add_syntheric_trend(generated_arma_sample_df)