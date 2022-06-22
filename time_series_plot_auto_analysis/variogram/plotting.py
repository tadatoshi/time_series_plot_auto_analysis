import numpy as np
import pandas as pd


def plot_and_save(data_frame: pd.DataFrame, file_path, xlabel='Lag k', ylabel='Gk', x_max=None, y_max=None):

    if x_max is None:
        x_max = len(data_frame)
    if y_max is None:
        y_max = np.ceil(data_frame.max().values[0])

    data_frame.plot(marker='o', xlabel=xlabel, ylabel=ylabel, legend=None,
                    xlim=([1, x_max]), ylim=([1, y_max]),
                    xticks=np.arange(1, x_max+1), yticks=np.arange(1, y_max+1, 0.5)
                    ).get_figure().savefig(file_path)
