import os
from time_series_plot_auto_analysis.analysis import StationarityDetection
from time_series_plot_auto_analysis.categories.stationarity import Stationarity


class TestAnalysis:
    """
    Test default implementation, i.e. the default keras model.
    """

    def test_train_model_and_save_in_saved_model_format(self):

        plots_directory_path = os.path.abspath(os.path.join(__file__, '../../../data/plots/synthetic_variograms'))
        models_path = os.path.abspath(os.path.join(__file__, '../../../tensorflow_models/saved_model/model_1'))

        stationarity_detection = StationarityDetection()
        original_model = stationarity_detection.train_model_and_save(plots_directory_path=plots_directory_path,
                                                                     models_path=models_path)
        loaded_model = stationarity_detection._load_model(models_path=models_path)

        assert original_model.summary() == loaded_model.summary()

    def test_stationarity_detaction(self):

        prediction_plots_directory_path = os.path.abspath(
            os.path.join(__file__, '../../../data/plots/synthetic_variograms_for_prediction'))
        prediction_plot_file_name = os.listdir(os.path.join(prediction_plots_directory_path,
                                                            Stationarity.stationary.value))[0]
        prediction_plot_file_path = os.path.join(prediction_plots_directory_path,
                                                 Stationarity.stationary.value,
                                                 prediction_plot_file_name)
        models_path = os.path.abspath(os.path.join(__file__, '../../../tensorflow_models/saved_model/model_1'))

        stationarity_detection = StationarityDetection()
        actual_stationarity = stationarity_detection.detect(models_path=models_path,
                                                            prediction_plot_file_path=prediction_plot_file_path)

        expected_stationarity = Stationarity.nonstationary

        assert actual_stationarity == expected_stationarity
