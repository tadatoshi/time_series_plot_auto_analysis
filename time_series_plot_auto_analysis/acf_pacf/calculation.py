import numpy as np
import pandas as pd


def add_syntheric_trend(generated_arma_sample_df: pd.DataFrame, trend_span_ratio=4):
    max = generated_arma_sample_df.max().values[0]
    min = generated_arma_sample_df.min().values[0]
    trend_span = (max - min) * trend_span_ratio
    trend = np.linspace(0, trend_span, len(generated_arma_sample_df))
    trend_df = pd.DataFrame(trend)
    return generated_arma_sample_df + trend_df
