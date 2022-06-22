import pytest
from pandas.testing import assert_frame_equal
from hypothesis import given, strategies as st
from time_series_plot_auto_analysis.variogram.calculation import calculate_variogram
import pandas as pd
from pandas.core.frame import DataFrame


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

    @pytest.mark.parametrize(
        "data, result",
        [
            ([1.0, 2.0, 4.0, 7.0], [1.0, 2.0])
        ]
    )
    def test_lags_two(self, data, result):

        expected_variogram = pd.DataFrame(result, columns=['g_k'])
        expected_variogram.index += 1

        data_df = pd.DataFrame(data, columns=['temperature'])
        actual_variogram = calculate_variogram(data_df, lags=2)

        assert_frame_equal(actual_variogram, expected_variogram)

    # @given(data=st.from_type(DataFrame), lags=st.just(20))
    # def test_fuzz_calculate_variogram(self, data, lags):
    #     calculate_variogram(
    #         data=data, lags=lags
    #     )
