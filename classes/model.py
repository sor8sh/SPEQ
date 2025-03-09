import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import metrics
from classes.Utils import Utils
from tensorflow.keras.callbacks import EarlyStopping


class SPEQ:
    def __init__(self, data_addr, output_addr, train_to_test='2,1', log=True, dataset_addr=None):
        self.dataset_address = f'{data_addr}/data.csv' if dataset_addr is None else dataset_addr
        self.out_dir = output_addr
        self.log = log
        self.train_to_test = list(map(int, train_to_test.split(',')))
        self.Y, self.X = None, None
        self.x_train, self.y_train, self.features_train = None, None, None
        self.x_test, self.y_test, self.features_test = None, None, None
        self.test_n, self.max_len = 0, 0
        self.spec_num, self.scan_num_test = None, None
        self.results = np.asarray([0])

    def model(self, load_w=False, batch=10, epoch=1, verbose=1, validation_split=.3, save_w=True, w_id=0):
        if load_w:
            Utils.log(self.log, "Loading the model:\n")
            return tf.keras.models.load_model(load_w)

        # Building the model
        # Input layers
        input_peaks = tf.keras.Input(shape=(self.max_len,), dtype="float32", name='peaks')
        input_features = tf.keras.Input(shape=(2,), dtype="float32", name='features')

        # Embedding and processing the input_peaks
        x_embed = layers.Embedding(input_dim=101, output_dim=64, input_length=self.max_len)(input_peaks)
        x_embed = layers.Dropout(0.3)(x_embed)

        x_peaks = layers.Conv1D(filters=128, kernel_size=11, strides=5, activation='relu')(x_embed)
        x_peaks = layers.MaxPooling1D()(x_peaks)
        x_peaks = layers.Dropout(0.3)(x_peaks)

        x_peaks = layers.Conv1D(filters=128, kernel_size=51, strides=10, activation='relu')(x_peaks)
        x_peaks = layers.MaxPooling1D()(x_peaks)
        x_peaks = layers.Dropout(0.3)(x_peaks)

        x_peaks = layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu')(x_peaks)
        x_peaks = layers.GlobalMaxPooling1D()(x_peaks)

        # Merge all available features into a single large vector via concatenation
        x = layers.concatenate([input_features, x_peaks], axis=1)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        predictions = layers.Dense(1, activation='sigmoid', name='predictions')(x)

        model = tf.keras.Model(inputs=[input_peaks, input_features], outputs=[predictions])
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Training the model
        Utils.log(self.log, "Training the model:\n")
        model.fit(
            {'peaks': self.x_train, 'features': self.features_train}, {'predictions': self.y_train},
            batch_size=batch, epochs=epoch, verbose=verbose, validation_split=validation_split, callbacks=[early_stop])
        if save_w: model.save(f'{self.out_dir}/weights/weights_{w_id}.h5')

        return model

    def load_data(self):
        Utils.log(self.log, 'Loading Dataset...\t')
        x_0, y_0, x_1, y_1, self.max_len = Utils.load_dataset(self.dataset_address)
        Utils.log(self.log, 'Done\n')

        train_portion_0 = np.ceil(self.train_to_test[0] * len(y_0[0]) / sum(self.train_to_test) / 10) * 10
        train_portion_1 = np.ceil(self.train_to_test[0] * len(y_1[0]) / sum(self.train_to_test) / 10) * 10
        self.test_n = int(len(x_0[0]) + len(x_1[0]) - train_portion_0 - train_portion_1)

        Utils.log(self.log, 'Generating Seeds...\t')
        x_train_str, y_train_biased, x_test_str, y_test_biased = Utils.set_seeds(x_0, y_0, x_1, y_1, train_portion_0, train_portion_1)
        Utils.log(self.log, 'Done\n')

        Utils.log(self.log, 'Converting Samples...\t')
        x_train_biased, x_test_biased = Utils.str2int(x_train_str, self.max_len, pad=0.0), Utils.str2int(x_test_str, self.max_len, pad=0.0)
        Utils.log(self.log, 'Done\n')

        # randomize training and test set (ones and zeros)
        x_train, features_train, y_train = Utils.get_training_set(x_train_biased, y_train_biased)
        self.scan_num_test, x_test, features_test, y_test = Utils.get_test_set(x_test_biased, y_test_biased, self.test_n)

        # selecting training, validation, and test set
        train_n = len(y_train)
        self.x_train, self.y_train = np.asarray(x_train[:train_n]), np.asarray(y_train[:train_n])
        self.x_test, self.y_test = np.asarray(x_test[:self.test_n]), np.asarray(y_test[:self.test_n])
        self.features_train = np.asarray(features_train[:train_n], dtype="float32")
        self.features_test = np.asarray(features_test[:self.test_n], dtype="float32")

    def test(self, model, verbose=1):
        Utils.log(self.log, "Testing the model:\n")
        self.X, features, self.Y, self.spec_num = self.x_test, self.features_test, self.y_test, self.scan_num_test
        score = model.evaluate({'peaks': self.X, 'features': features}, {'predictions': self.Y}, batch_size=32, verbose=verbose)

        print(f'Test loss: {score[0]}')
        print(f'Test accuracy: {score[1]}')
        print(f'{int(np.sum(self.y_test))} Positive samples | {int(len(self.y_test) - np.sum(self.y_test))} Negative samples')

        self.results = model.predict({'peaks': self.X, 'features': features}, batch_size=32, verbose=verbose)

    # region Metrics & Results
    def get_auc(self):
        real_results = Utils.get_real_results(self.results)
        return metrics.roc_auc_score(self.Y, real_results)

    def get_roc_plot(self, addr=None, p_id=0):
        if addr is None:
            os.makedirs(f'{self.out_dir}/plots', exist_ok=True)
            addr = f'{self.out_dir}/plots/ROC_curve_{p_id}.png'
        real_results = Utils.get_real_results(self.results)
        fpr, tpr, _ = metrics.roc_curve(self.Y, real_results)

        # plot
        legend = 'AUC: ' + str(-1 * round(-1 * np.trapz(tpr, fpr), 4))
        plt.plot(fpr, tpr, marker='', label=legend)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(addr)
        plt.close()

    def get_confusion_matrix(self):
        y_hat = (self.results >= 0.5).astype(int)
        return metrics.confusion_matrix(self.Y, y_hat).ravel()

    def get_results(self, addr=None, l_id=0):
        if addr is None:
            os.makedirs(f'{self.out_dir}/results', exist_ok=True)
            addr = f'{self.out_dir}/results/results_{l_id}.csv'
        scores = Utils.get_real_results(self.results)
        df = pd.DataFrame({"Spectrum": self.spec_num, "Label": self.Y, "Score": scores.flatten()})
        df.to_csv(addr, index=False)
    # endregion
