from tensorflow import keras
import numpy as np


class Model:

    def __init__(self):
        self.model = keras.models.load_model('./models/best_model5.h5')

        self.predicted_class = None
        self.probabilities = None

    def predict(self, image: np.array):
        if self.predicted_class is None:
            self.probabilities = self.model.predict(image)
            self.predicted_class = np.argmax(self.probabilities, axis=-1)[0]

        return self.predicted_class

    def get_probabilities(self):
        return self.probabilities
