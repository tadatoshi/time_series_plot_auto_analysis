import os
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample

from time_series_plot_auto_analysis.plotting.plotting import PlottingUtil


class AbstractSyntheticPlot(ABC):

    def generate(self, plots_directory_path=None, number_of_plots=100,
                 lags=30, arparams=[], maparams=[], number_of_samples=100):

        if plots_directory_path is None:
            plots_directory_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '../../plots'))

        arparams = np.array(arparams)
        full_arparams = np.r_[1, -arparams]
        maparams = np.array(maparams)
        full_maparams = np.r_[1, -maparams]
        arima_order = (len(arparams), 0, len(maparams))

        current_time_string = self._current_time_string()

        directory_name =\
            f"{self._directory_file_name_prefix()}_{current_time_string}"
        directory_path = os.path.abspath(os.path.join(
            plots_directory_path, directory_name))
        os.mkdir(directory_path)

        plotting_util = self._plotting_util()

        for i in range(number_of_plots):
            generated_arma_sample = arma_generate_sample(
                full_arparams, full_maparams, number_of_samples)
            generated_arma_sample_df = pd.DataFrame(generated_arma_sample)
            data_to_plot = self._additional_calculation(
                generated_arma_sample_df, lags=lags, arima_order=arima_order)
            file_name = f"{self._directory_file_name_prefix()}" + \
                        f"_{current_time_string}_{i+1}.png"
            file_path = os.path.join(directory_path, file_name)
            plotting_util.plot_and_save(data_to_plot, file_path,
                            axis_off=True, save_without_displaying_plot=True)

    def _current_time_string(self):
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def _plotting_util(self):
        return PlottingUtil()

    @abstractmethod
    def _directory_file_name_prefix(self):
        pass

    @abstractmethod
    def _additional_calculation(self, generated_arma_sample_df, **kwargs):
        pass
