from time_series_plot_auto_analysis.acf_pacf.plotting import AcfPlottingUtil, PacfPlottingUtil
from time_series_plot_auto_analysis.plotting.synthetic import AbstractSyntheticPlot


class SyntheticAcfPacfPlot(AbstractSyntheticPlot):

    def generate(self, plots_directory_path=None, number_of_plots=100, lags=30,
                 arparams=[], maparams=[], number_of_samples=100):

        current_time_string = self._current_time_string()

        synthetic_acf_plot = self._syntheric_acf_plot()
        synthetic_acf_plot.set_current_time_string(current_time_string)
        synthetic_acf_plot.generate(plots_directory_path=plots_directory_path,
                number_of_plots=number_of_plots, lags=lags,
                arparams=arparams, maparams=maparams,
                number_of_samples=number_of_samples)

        synthetic_pacf_plot = self._syntheric_pacf_plot()
        synthetic_pacf_plot.set_current_time_string(current_time_string)
        synthetic_pacf_plot.generate(plots_directory_path=plots_directory_path,
                number_of_plots=number_of_plots, lags=lags,
                arparams=arparams, maparams=maparams,
                number_of_samples=number_of_samples)

    def _syntheric_acf_plot(self):
        return SyntheticAcfPlot()

    def _syntheric_pacf_plot(self):
        return SyntheticPacfPlot()

    # Not used
    def _directory_file_name_prefix(self):
        pass

    # Not used
    def _additional_calculation(self, generated_arma_sample_df, lags=15):
        return generated_arma_sample_df

class SyntheticAcfPlot(AbstractSyntheticPlot):

    def set_current_time_string(self, current_time_string):
        self._assigned_current_time_string = current_time_string

    def _plotting_util(self):
        return AcfPlottingUtil()

    def _current_time_string(self):
        return self._assigned_current_time_string

    def _directory_file_name_prefix(self):
        return "synthetic_acf"

    def _additional_calculation(self, generated_arma_sample_df, lags=15):
        return generated_arma_sample_df

class SyntheticPacfPlot(AbstractSyntheticPlot):

    def set_current_time_string(self, current_time_string):
        self._assigned_current_time_string = current_time_string

    def _plotting_util(self):
        return PacfPlottingUtil()

    def _current_time_string(self):
        return self._assigned_current_time_string

    def _directory_file_name_prefix(self):
        return "synthetic_pacf"

    def _additional_calculation(self, generated_arma_sample_df, lags=15):
        return generated_arma_sample_df
