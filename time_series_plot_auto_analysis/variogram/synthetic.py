from time_series_plot_auto_analysis.plotting.synthetic import AbstractSyntheticPlot
from time_series_plot_auto_analysis.variogram.calculation import calculate_variogram


class SyntheticVariogramPlot(AbstractSyntheticPlot):

    def _directory_file_name_prefix(self):
        return "synthetic_variogram"

    def _additional_calculation(self, generated_arma_sample_df, **kwargs):
        return calculate_variogram(generated_arma_sample_df,
                                   lags=kwargs['lags'])
