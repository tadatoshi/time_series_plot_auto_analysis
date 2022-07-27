import tensorflow as tf
from tensorflow import keras
from time_series_plot_auto_analysis.analysis import StationarityDetection


class AcfPacfStationarityDetection(StationarityDetection):

    def _train_model(self, plot_image_height, plot_image_width, plot_image_color,
                     train_dataset, validation_dataset, epochs=3):
        """
        Defines unique model for ACF and PACF plot and train it.
        """

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
