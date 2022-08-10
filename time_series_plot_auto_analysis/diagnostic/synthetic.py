from time_series_plot_auto_analysis.diagnostic.plotting import QqPlotUtil
from time_series_plot_auto_analysis.plotting.synthetic import AbstractSyntheticPlot
from time_series_plot_auto_analysis.diagnostic.calculation import fit_model_and_get_residual


class SyntheticQqPlot(AbstractSyntheticPlot):

    def _plotting_util(self):
        return QqPlotUtil()

    def _directory_file_name_prefix(self):
        return "synthetic_qqplot"

    def _additional_calculation(self, generated_arma_sample_df, **kwargs):
        return fit_model_and_get_residual(generated_arma_sample_df,
                                          kwargs['arima_order'])
