import pandas as pd
import statsmodels.api as sm
from time_series_plot_auto_analysis.plotting.plotting import PlottingUtil


class AcfPlottingUtil(PlottingUtil):

    def _arguments(self, data_frame, xlabel='x', ylabel='y', x_max=None, y_max=None, lags=None):
        arguments = {'lags': lags}
        return arguments

    def _plot_and_get_figure(self, data_frame: pd.DataFrame, arguments):
        return sm.graphics.tsa.plot_acf(data_frame, **arguments)

class PacfPlottingUtil(PlottingUtil):

    def _arguments(self, data_frame, xlabel='x', ylabel='y', x_max=None, y_max=None, lags=None):
        arguments = {'lags': lags}
        return arguments

    def _plot_and_get_figure(self, data_frame: pd.DataFrame, arguments):
        return sm.graphics.tsa.plot_pacf(data_frame, **arguments)
