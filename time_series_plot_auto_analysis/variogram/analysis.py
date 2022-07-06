import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from time_series_plot_auto_analysis.categories.stationarity import Stationarity


class StationarityDetection:

    def __init__(self):
        pass

    def train_model_and_save(self, plots_directory_path=None, models_path=None, validation_split=0.2, batch_size=32,
                             epochs=3):

        plot_image_height, plot_image_width, plot_image_color = self._get_plot_image_height_width_and_color(
                                                                        plots_directory_path=plots_directory_path)

        train_dataset, validation_dataset = self._get_datasets(batch_size, plot_image_height, plot_image_width,
                                                               plots_directory_path, validation_split)

        model = self._train_model(plot_image_height, plot_image_width, plot_image_color,
                                  train_dataset, validation_dataset, epochs=epochs)

        model.save(models_path)

        return model

    def detect(self, models_path=None, prediction_plot_file_path=None):

        plot_image = cv2.imread(prediction_plot_file_path)
        plot_image_height = plot_image.shape[0]
        plot_image_width = plot_image.shape[1]
        prediction_plot = keras.utils.load_img(
            prediction_plot_file_path, target_size=(plot_image_height, plot_image_width)
        )
        prediction_plot_array = tf.keras.utils.img_to_array(prediction_plot)
        prediction_plot_array = tf.expand_dims(prediction_plot_array, 0)

        model = self._load_model(models_path=models_path)
        prediction_plot_predictions = model.predict(prediction_plot_array)
        prediction_plot_prediction_score = tf.nn.softmax(prediction_plot_predictions[0])

        stationarity_enum_list = [stationarity for stationarity in Stationarity]
        return stationarity_enum_list[np.argmax(prediction_plot_prediction_score)]

    def _load_model(self, models_path=None):
        return keras.models.load_model(models_path)

    def _get_plot_image_height_width_and_color(self, plots_directory_path=None):
        sample_plot_file_name = os.listdir(os.path.join(plots_directory_path, Stationarity.stationary.value))[0]
        sample_plot_file_path = os.path.join(plots_directory_path, Stationarity.stationary.value, sample_plot_file_name)
        plot_image = cv2.imread(sample_plot_file_path)
        return plot_image.shape

    def _get_datasets(self, batch_size, plot_image_height, plot_image_width, plots_directory_path, validation_split):
        train_dataset = keras.utils.image_dataset_from_directory(
            plots_directory_path,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(plot_image_height, plot_image_width),
            batch_size=batch_size)
        validation_dataset = keras.utils.image_dataset_from_directory(
            plots_directory_path,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(plot_image_height, plot_image_width),
            batch_size=batch_size)
        # Configure the dataset for performance
        train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_dataset, validation_dataset

    def _train_model(self, plot_image_height, plot_image_width, plot_image_color,
                     train_dataset, validation_dataset, epochs=3):

        num_classes = 2
        model = keras.Sequential([
            keras.layers.Rescaling(1. / 255, input_shape=(plot_image_height, plot_image_width, plot_image_color)),
            keras.layers.Conv2D(16, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes)
        ])

        model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs
        )

        return model
