import pytest
from time_series_plot_auto_analysis.variogram.calculation import calculate_variogram
import pandas as pd
from pandas.util.testing import assert_frame_equal


class TestVariogram:

    @pytest.fixture()
    def furnace_data(self):
        data_df = pd.read_csv("data/temperature_readings_from_a_ceramic_furnace.csv", header=None,
                              names=["temperature"])
        yield data_df

    @pytest.mark.skip(reason="Implement calculate_variogram function")
    def test_one_time_unit_apart(self, furnace_data):

        expected_variogram_g_k = pd.DataFrame([1.0], columns=['g_k'])

        actual_variogram_g_k = calculate_variogram(furnace_data, lags=1)

        assert assert_frame_equal(actual_variogram_g_k, expected_variogram_g_k)
