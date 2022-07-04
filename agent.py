from tensorflow import keras
import os


class Agent:

    def __init__(self, layers, loss_fn, optimizer, discount_factor, file=None):
        # Save important params as class attributes
        self.hidden = [n_units for n_units, _ in layers[1:-1]]
        self.n_outputs = layers[-1][0]
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.file = file
        # Build model if no file is given, else load model from file
        if file is None:
            self.model = keras.Sequential()
            self.model.add(keras.layers.InputLayer(input_shape=(layers[0],)))
            for n_units, activation in layers[1:]:
                self.model.add(keras.layers.Dense(n_units, activation=activation))
            self.model.compile(optimizer, loss_fn)
        else:
            self.model = keras.models.load_model(file)
            print('Model loaded from file.')

    def save(self, file_name):
        # Create necessary directories if they don't exist
        if not os.path.isdir('./models'):
            os.mkdir('./models')
        path = './models/' + file_name
        if not os.path.isdir(path):
            os.mkdir(path)
        # Create file name using layer sizes
        file_path = path + '/' + file_name
        for h in self.hidden:
            file_path += '_{}'.format(h)
        file_path += '.h5'
        # Save model to file
        self.model.save(file_path)
        print('Model saved.')
