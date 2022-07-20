import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PlottingUtil:

    def plot_and_save(self, data_frame: pd.DataFrame, file_path, xlabel='x', ylabel='y', x_max=None, y_max=None,
                      marker_on=False, axis_off=False, save_without_displaying_plot=False, lags=None):

        arguments = self._arguments(data_frame, xlabel=xlabel, ylabel=ylabel, x_max=x_max, y_max=y_max,
                                    lags=lags)
        if marker_on:
            arguments['marker'] = 'o'

        fig = self._plot_and_get_figure(data_frame, arguments)
        if axis_off:
            plt.axis('off')
        if save_without_displaying_plot:
            plt.close(fig)
        fig.savefig(file_path)

    def _arguments(self, data_frame, xlabel='x', ylabel='y', x_max=None, y_max=None, lags=None):
        if x_max is None:
            x_max = len(data_frame)
        if y_max is None:
            y_max = np.ceil(data_frame.max().values[0])
        arguments = {'xlabel': xlabel, 'ylabel': ylabel, 'legend': None,
                     'xlim': ([1, x_max]), 'ylim': ([1, y_max]),
                     'xticks': np.arange(1, x_max + 1), 'yticks': np.arange(1, y_max + 1, 0.5)}
        return arguments

    def _plot_and_get_figure(self, data_frame: pd.DataFrame, arguments):
        return data_frame.plot(**arguments).get_figure()
