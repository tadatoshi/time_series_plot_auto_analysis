import pandas as pd
from statsmodels.graphics.api import qqplot
from time_series_plot_auto_analysis.plotting.plotting import PlottingUtil


class QqPlotUtil(PlottingUtil):

    def _arguments(self, data_frame, xlabel='x', ylabel='y', x_max=None, y_max=None, lags=None):
        return {}

    def _plot_and_get_figure(self, data_frame: pd.DataFrame, arguments):
        return qqplot(data_frame, line="q", fit=True)
