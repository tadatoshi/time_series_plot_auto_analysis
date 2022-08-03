import numpy as np
import pandas as pd


def add_syntheric_trend(generated_arma_sample_df: pd.DataFrame,
                        trend_span_ratio=4):
    max = generated_arma_sample_df.max().values[0]
    min = generated_arma_sample_df.min().values[0]
    trend_span = (max - min) * trend_span_ratio
    trend = np.linspace(0, trend_span, len(generated_arma_sample_df))
    trend_df = pd.DataFrame(trend)
    return generated_arma_sample_df + trend_df


def add_synthetic_seasonality(generated_arma_sample_df: pd.DataFrame,
                              regular_period_p=15, wave_number_n=3):
    seasonality = generate_fourier_basis(
        np.arange(len(generated_arma_sample_df)),
        p=regular_period_p, n=wave_number_n)
    seasonality_df = pd.DataFrame(seasonality)
    return generated_arma_sample_df + seasonality_df


def generate_fourier_basis(t, p=15, n=3):
    """

    :param t: time - array like
    :param p: Regular period of the time series
    :param n: Wave number - n of 2 * pi * n * t
    :return: fourier basis function data
    """
    x = 2 * np.pi * n * t / p
    return np.sum([np.cos(x), np.sin(x)], axis=0)
    # return np.concatenate((np.cos(x), np.sin(x)), axis=1)
