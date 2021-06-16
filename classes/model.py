import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
import os


def str2int(str_samples, max_len, pad=0.0):
    int_samples = [[], [], []]
    if type(pad) == float:
        for i in range(len(str_samples[0])):
            temp = str_samples[1][i].split()
            row = [float(_) for _ in temp]
            if len(row) < max_len:
                left = int(np.floor((max_len - len(row)) / 2))
                right = int(np.ceil((max_len - len(row)) / 2))
                row = [pad] * left + row + [pad] * right
            int_samples[0].append(str_samples[0][i])
            int_samples[1].append(row)
            int_samples[2].append((str_samples[2][i]))
    elif pad == 'mirror':
        for i in range(len(str_samples[0])):
            temp = str_samples[1][i].split()
            row = [float(_) for _ in temp]
            if len(row) < max_len:
                left = int(np.floor((max_len - len(row)) / 2))
                right = int(np.ceil((max_len - len(row)) / 2))
                row = np.pad(row, (left, right), mode='reflect')
            int_samples[0].append(str_samples[0][i])
            int_samples[1].append(row)
            int_samples[2].append((str_samples[2][i]))
    return int_samples


def get_training_set(biased_x, biased_y):
    temp = list(zip(biased_x[0], biased_x[1], biased_x[2], biased_y[1]))
    random.shuffle(temp)
    scan_num, x, features, y = zip(*temp)
    return scan_num, x, features, y


def get_test_set(biased_x, biased_y, test_n):
    temp = list(zip(biased_x[0], biased_x[1], biased_x[2], biased_y[1]))
    random.shuffle(temp)
    scan_num, x, features, y = zip(*temp)
    return scan_num[:test_n], x[:test_n], features[:test_n], y[:test_n]


def get_real_results(raw_results):
    real_results = np.zeros(raw_results.shape)
    for i in range(raw_results.shape[0]):
        if raw_results[i][0] == 1:
            real_results[i][0] = 20
            continue
        elif raw_results[i][0] == 0:
            real_results[i][0] = -20
            continue
        real_results[i][0] = np.log(raw_results[i][0] / (1 - raw_results[i][0]))
    return real_results


def roc_curve(y, y_hat, n, p, test_n):
    fpr, tpr, precision, recall = [], [], [], []
    # print(min(y_hat), max(y_hat))
    thresholds = np.arange(min(y_hat) - 0.1, max(y_hat) + 0.1, .1)
    for thresh in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(test_n):
            if y_hat[i][0] >= thresh:
                if y[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if y[i] == 0:
                    tn += 1
                else:
                    fn += 1
        if tp + fp > 0:
            precision.append(tp / (tp + fp))
        else:
            precision.append(precision[-1])
        if tp + fn > 0:
            recall.append(tp / (tp + fn))
        else:
            recall.append(recall[-1])
        fpr.append(fp / float(n))
        tpr.append(tp / float(p))
    return fpr, tpr, precision, recall


def confusion_matrix(y, y_pred, thresh, test_n):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(test_n):
        y_hat = 0
        if y_pred[i][0] >= thresh:
            y_hat = 1

        if y_hat == y[i]:
            if y[i] == 0:
                tn += 1
            else:
                tp += 1
        else:
            if y[i] == 0:
                fp += 1
            else:
                fn += 1
    return tp, fp, tn, fn


def set_seeds(x_0, y_0, x_1, y_1, train_portion_0, train_portion_1):
    good_seeds_train = random.sample(range(0, len(x_1[0])), int(train_portion_1))  # good samples training seeds
    good_seeds_test = [i for i in range(len(x_1[0])) if i not in good_seeds_train]  # good samples test seeds
    bad_seeds_train = random.sample(range(0, len(x_0[0])), int(train_portion_0))  # bad samples training seeds
    bad_seeds_test = [i for i in range(len(x_0[0])) if i not in bad_seeds_train]  # bad samples test seeds
    train_length = min(len(good_seeds_train), len(bad_seeds_train))

    x_train, y_train = [[], [], []], [[], []]
    x_test, y_test = [[], [], []], [[], []]
    for i in range(train_length):
        x_train[0].append(x_1[0][good_seeds_train[i]])
        x_train[1].append(x_1[1][good_seeds_train[i]])
        x_train[2].append(x_1[2][good_seeds_train[i]])
        y_train[0].append(y_1[0][good_seeds_train[i]])
        y_train[1].append(y_1[1][good_seeds_train[i]])
    for i in range(len(good_seeds_test)):
        x_test[0].append(x_1[0][good_seeds_test[i]])
        x_test[1].append(x_1[1][good_seeds_test[i]])
        x_test[2].append(x_1[2][good_seeds_test[i]])
        y_test[0].append(y_1[0][good_seeds_test[i]])
        y_test[1].append(y_1[1][good_seeds_test[i]])
    for i in range(train_length):
        x_train[0].append(x_0[0][bad_seeds_train[i]])
        x_train[1].append(x_0[1][bad_seeds_train[i]])
        x_train[2].append(x_0[2][bad_seeds_train[i]])
        y_train[0].append(y_0[0][bad_seeds_train[i]])
        y_train[1].append(y_0[1][bad_seeds_train[i]])
    for i in range(len(bad_seeds_test)):
        x_test[0].append(x_0[0][bad_seeds_test[i]])
        x_test[1].append(x_0[1][bad_seeds_test[i]])
        x_test[2].append(x_0[2][bad_seeds_test[i]])
        y_test[0].append(y_0[0][bad_seeds_test[i]])
        y_test[1].append(y_0[1][bad_seeds_test[i]])

    return x_train, y_train, x_test, y_test


class DeSQ:
    def __init__(self, output_addr, train_to_test='2,1', log=True, dataset_addr=None):
        self.out_dir = output_addr
        if dataset_addr is None:
            self.dataset_address = '{}/data.csv'.format(output_addr)
        else:
            self.dataset_address = dataset_addr
        self.log = log

        temp = train_to_test.split(',')
        self.train_to_test = [int(temp[0]), int(temp[1])]

        self.Y, self.X = [0], [0]
        self.N, self.P = 0, 0
        self.test_n = 0
        self.results = np.asarray([0])
        self.spec_num = np.asarray([0])

    def load_dataset(self):
        x_1, y_1 = [[], [], []], [[], []]
        x_0, y_0 = [[], [], []], [[], []]
        max_len = 0

        with open(self.dataset_address, 'r') as f:
            lines = f.readlines()
            lines.pop(0)
            for line in lines:
                # Extract data
                elements = line.split(',')
                scan = int(elements[0])
                label = float(elements[1])
                features = [elements[2], elements[3]]

                # good samples
                if label:
                    # y = scan number, label
                    y_1[0].append(scan)
                    y_1[1].append(1.0)

                    # x = scan number, 1D sequence, features
                    x_1[0].append(scan)
                    x_1[1].append(elements[4])
                    x_1[2].append([int(features[0]), float(features[1])])

                    # maximum length
                    if len(elements[4].split()) > max_len:
                        max_len = len(elements[4].split())

                # bad samples
                else:
                    # y = scan number, label
                    y_0[0].append(scan)
                    y_0[1].append(0.0)

                    # x = scan number, 1D sequence, features
                    x_0[0].append(scan)
                    x_0[1].append(elements[4])
                    x_0[2].append([int(features[0]), float(features[1])])

                    # maximum length
                    if len(elements[4].split()) > max_len:
                        max_len = len(elements[4].split())

        return x_0, y_0, x_1, y_1, max_len

    def get_auc(self):
        real_results = get_real_results(self.results)
        fpr, tpr, precision, recall = roc_curve(self.Y, real_results, self.N, self.P, self.test_n)

        # calculate AUC
        auc = -1 * np.trapz(tpr, fpr)
        return auc

    def get_roc_plot(self, addr=None, p_id=0):
        real_results = get_real_results(self.results)
        fpr, tpr, precision, recall = roc_curve(self.Y, real_results, self.N, self.P, self.test_n)

        # calculate AUC
        auc = -1 * np.trapz(tpr, fpr)

        # plot
        legend = 'AUC: ' + str(-1 * round(auc, 4))
        plt.plot(fpr, tpr, marker='', label=legend)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        if addr is not None:
            plt.savefig(addr)
        else:
            try:
                os.mkdir('{}/plots'.format(self.out_dir))
            except FileExistsError:
                pass
            plt.savefig('{}/plots/ROC_curve_{}.png'.format(self.out_dir, p_id))
        plt.close()

    def get_confusion_matrix(self):
        tp, fp, tn, fn = confusion_matrix(self.Y, self.results, 0.5, self.test_n)
        return tp, fp, tn, fn

    def get_results(self, addr=None, l_id=0):
        real_results = get_real_results(self.results)
        log_file, fp_file, fn_file = [], [], []
        for i in range(self.test_n):
            log = str(self.spec_num[i]) + ' ' + str(self.Y[i]) + ' ' + str(real_results[i][0])
            log_file.append(log)

        if addr is not None:
            with open(addr, 'w') as f:
                f.write('Scan Number, Label, Score')
                for line in log_file:
                    f.write(line + '\n')
        else:
            try:
                os.mkdir('{}/results'.format(self.out_dir))
            except FileExistsError:
                pass
            with open('{}/results/results_{}.txt'.format(self.out_dir, l_id), 'w') as f:
                f.write('Scan Number, Label, Score')
                for line in log_file:
                    f.write(line + '\n')

    def run(self, mode, batch_n=10, epoch_n=1, verbose=1, validation_split=.3,
            load_weights=False, save_weights=True, weight_id=0):
        # Load Dataset
        if self.log:
            print('Loading Dataset...\t', end="", flush=True)
        x_0, y_0, x_1, y_1, max_len = self.load_dataset()
        if self.log:
            print('Done')

        train_portion_0 = np.ceil(self.train_to_test[0] * len(y_0[0]) / sum(self.train_to_test)) // 10 * 10
        train_portion_1 = np.ceil(self.train_to_test[0] * len(y_1[0]) / sum(self.train_to_test)) // 10 * 10
        self.test_n = int(len(x_0[0]) + len(x_1[0]) - train_portion_0 - train_portion_1)

        if self.log:
            print('Generating Seeds...\t', end="", flush=True)
        x_train_str, y_train_biased, x_test_str, y_test_biased = set_seeds(x_0, y_0, x_1, y_1,
                                                                           train_portion_0, train_portion_1)
        if self.log:
            print('Done')

        if self.log:
            print('Converting Samples...\t', end="", flush=True)
        x_train_biased = str2int(x_train_str, max_len, pad=0.0)
        x_test_biased = str2int(x_test_str, max_len, pad=0.0)
        if self.log:
            print('Done')

        # randomize training and test set (ones and zeros)
        scan_num_train, x_train, features_train, y_train = get_training_set(x_train_biased, y_train_biased)
        scan_num_test, x_test, features_test, y_test = get_test_set(x_test_biased, y_test_biased, self.test_n)

        # selecting training, validation, and test set
        train_n = len(y_train)
        x_train, y_train = np.asarray(x_train[:train_n]), np.asarray(y_train[:train_n])
        x_test, y_test = np.asarray(x_test[:self.test_n]), np.asarray(y_test[:self.test_n])
        features_train = np.asarray(features_train[:train_n], dtype="float32")
        features_test = np.asarray(features_test[:self.test_n], dtype="float32")
        self.P = np.sum(y_test)
        self.N = len(y_test) - self.P

        # build the model
        if mode.lower() == 'train' or mode == '1':
            # Input layers
            input_peaks = tf.keras.Input(shape=(None,), dtype="int64", name='peaks')
            input_features = tf.keras.Input(shape=(2,), dtype="float32", name='features')

            # Embedding and processing the input_peaks
            x_embed = layers.Embedding(input_dim=101, output_dim=64, input_length=max_len)(input_peaks)
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

            # Building/Loading the model
            if not load_weights:
                model = tf.keras.Model(inputs=[input_peaks, input_features], outputs=[predictions])
            else:
                model = tf.keras.models.load_model(load_weights)

            # Compiling the model with Adam optimizer
            model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

            # Training the model
            if self.log:
                print("Training the model:")
            model.fit(
                {'peaks': x_train, 'features': features_train}, {'predictions': y_train},
                batch_size=batch_n, epochs=epoch_n, verbose=verbose, validation_split=validation_split)
            if save_weights:
                try:
                    os.mkdir('{}/weights'.format(self.out_dir))
                except FileExistsError:
                    pass
                model.save('{}/weights/weights_{}.h5'.format(self.out_dir, weight_id))
        elif mode.lower() == 'test' or mode == '2':
            if self.log:
                print("Loading the model:")
            model = tf.keras.models.load_model(load_weights)
        else:
            model = None
            print('Error: wrong mode')
            print('valid value for [mode]: (1 or train) (2 or test)')
            exit()

        # test the model
        if self.log:
            print("Testing the model:")
        self.X, features, self.Y, self.spec_num = x_test, features_test, y_test, scan_num_test
        score = model.evaluate({'peaks': self.X, 'features': features}, {'predictions': self.Y},
                               steps=self.test_n, verbose=verbose)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('{} Positive samples - {} Negative samples'.format(self.P, self.N))
        self.results = model.predict({'peaks': self.X, 'features': features}, steps=self.test_n)
        return self.results
