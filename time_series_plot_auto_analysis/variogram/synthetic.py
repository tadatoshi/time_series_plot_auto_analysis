import os
from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
from time_series_plot_auto_analysis.variogram.calculation import calculate_variogram
from time_series_plot_auto_analysis.variogram.plotting import plot_and_save


def generate_synthetic_variogram_plots(number_of_plots=100, variogram_lags=15,
                                       arparams=[], maparams=[], number_of_samples=100):

    arparams = np.array(arparams)
    arparams = np.r_[1, -arparams]
    maparams = np.array(maparams)
    maparams = np.r_[1, -maparams]

    current_time_string = datetime.now().strftime("%Y%m%d%H%M%S")
    directory_name = f"synthetic_variograms_{current_time_string}"
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../plots', directory_name))
    os.mkdir(directory_path)

    for i in range(number_of_plots):
        generated_arma_sample = arma_generate_sample(arparams, maparams, number_of_samples)
        generated_arma_sample_df = pd.DataFrame(generated_arma_sample)
        synthetic_variogram = calculate_variogram(generated_arma_sample_df, lags=variogram_lags)
        file_name = f"synthetic_variogram_{current_time_string}_{i+1}.png"
        file_path = os.path.join(directory_path, file_name)
        plot_and_save(synthetic_variogram, file_path, axis_off=True, save_without_displaying_plot=True)
