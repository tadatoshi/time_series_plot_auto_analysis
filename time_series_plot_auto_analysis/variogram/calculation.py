import pandas as pd


def calculate_variogram(data, lags=20):

    variogram_list = []

    one_lag_difference = data.diff().dropna()
    one_lag_difference_variance = one_lag_difference.var()

    for i in range(lags):
        lag_difference = data.diff(i + 1).dropna()
        lag_difference_variance = lag_difference.var()
        variogram = lag_difference_variance / one_lag_difference_variance
        variogram_list.append(variogram.iloc[0])

    variogram_df = pd.DataFrame(variogram_list, columns=['g_k'])
    variogram_df.index += 1
    return variogram_df
