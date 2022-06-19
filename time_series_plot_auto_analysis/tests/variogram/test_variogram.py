import pytest
from time_series_plot_auto_analysis.variogram.calculation import calculate_variogram
import pandas as pd
from pandas.testing import assert_frame_equal


class TestVariogram:

    @pytest.fixture()
    def furnace_data(self):
        data_df = pd.read_csv("data/temperature_readings_from_a_ceramic_furnace.csv", header=None,
                              names=["temperature"])
        yield data_df

    def test_one_time_unit_apart(self, furnace_data):

        expected_variogram = pd.DataFrame([1.0], columns=['g_k'])
        expected_variogram.index += 1

        actual_variogram = calculate_variogram(furnace_data, lags=1)

        assert_frame_equal(actual_variogram, expected_variogram)
