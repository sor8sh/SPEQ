import os
import numpy as np
import random
import sys


class Utils:
    @staticmethod
    def create_directory(path):
        try: os.makedirs(path, exist_ok=True)
        except PermissionError as e: raise PermissionError(f"Cannot create output directory at {path}.") from e

    @staticmethod
    def log(log, message):
        if log: print(message, end="", flush=True)

    @staticmethod
    def progress(log, message, current, total):
        if not log: return
        percentage = int((current / total) * 100)
        sys.stdout.write(f"\r{message}\t{percentage}% ({current}/{total})")
        if current+1 == total: sys.stdout.write(f"\r{message}\t100% ({total}/{total})\tDone!\n")

    @staticmethod
    def load_dataset(address):
        x_1, y_1 = [[], [], []], [[], []]
        x_0, y_0 = [[], [], []], [[], []]
        max_len = 0

        with open(address, 'r') as f:
            lines = f.readlines()
            lines.pop(0)
            for line in lines:
                # Extract data
                scan, label, charge, mz, peaks = line.split(',')
                max_len = max(len(peaks.split()), max_len)

                # x = scan number, 1D sequence, features
                # y = scan number, label
                if int(label):
                    y_1[0].append(scan)
                    y_1[1].append(1.0)
                    x_1[0].append(scan)
                    x_1[1].append(peaks)
                    x_1[2].append([int(charge), float(mz)])
                else:
                    y_0[0].append(scan)
                    y_0[1].append(0.0)
                    x_0[0].append(scan)
                    x_0[1].append(peaks)
                    x_0[2].append([int(charge), float(mz)])

        return x_0, y_0, x_1, y_1, max_len

    @staticmethod
    def str2int(str_samples, max_len, pad=0.0):
        int_samples = [[], [], []]
        for i in range(len(str_samples[0])):
            row = np.array(str_samples[1][i].split(), dtype=float)

            if len(row) < max_len:
                pad_width = (max_len - len(row)) // 2
                row = np.pad(row, (pad_width, max_len - len(row) - pad_width), mode='constant', constant_values=pad)

            int_samples[0].append(str_samples[0][i])
            int_samples[1].append(row)
            int_samples[2].append(str_samples[2][i])

        return int_samples

    @staticmethod
    def get_training_set(biased_x, biased_y):
        data = list(zip(biased_x[0], biased_x[1], biased_x[2], biased_y[1]))
        random.shuffle(data)
        _, x, features, y = zip(*data)
        return x, features, y

    @staticmethod
    def get_test_set(biased_x, biased_y, test_n):
        data = list(zip(biased_x[0], biased_x[1], biased_x[2], biased_y[1]))
        random.shuffle(data)
        scan_num, x, features, y = zip(*data)
        return scan_num[:test_n], x[:test_n], features[:test_n], y[:test_n]

    @staticmethod
    def get_real_results(raw_results):
        real_results = np.zeros(raw_results.shape)
        for i in range(raw_results.shape[0]):
            if raw_results[i][0] == 1: real_results[i][0] = 20
            elif raw_results[i][0] == 0: real_results[i][0] = -20
            else: real_results[i][0] = np.log(raw_results[i][0] / (1 - raw_results[i][0]))
        return real_results

    @staticmethod
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

    @staticmethod
    def get_one_d_seq(intensity, mz):
        intensity = (np.array(intensity) / max(intensity)) * 100
        peaks = list(map(lambda mz, i: (round(mz, 2), round(i, 2)), mz, intensity))
        peaks = [p for p in peaks if p[1] > 1]

        min_peak = int(peaks[0][0])
        max_peak = int(peaks[-1][0])
        seq = np.zeros(max_peak - min_peak + 1)

        for peak, intensity in peaks:
            index = int((peak + 0.4) / 1.0005079) - min_peak
            if 0 <= index < len(seq):
                seq[index] = max(seq[index], intensity)

        seq = (seq / seq.max()) * 100 if seq.max() > 0 else seq
        return seq.tolist()
