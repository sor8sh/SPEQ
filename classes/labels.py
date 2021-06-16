import operator
import csv
import numpy as np
import math
import matplotlib.pyplot as plt


def calculate_density(data):
    density = {}
    keys = []
    for _ in data[1]:
        if int(math.modf(_)[1]) in density:
            density[int(math.modf(_)[1])] += 1
        else:
            density[int(math.modf(_)[1])] = 1
            keys.append(int(math.modf(_)[1]))
    return density, keys


def get_xy(density, keys):
    index = np.arange(min(keys), max(keys) + 1)
    xy = []
    for i in range(len(index)):
        if index[i] in density:
            xy.append((index[i], density[index[i]]))
        else:
            xy.append((index[i], 0))
    return xy


def separate_x_y(xy):
    x = []
    y = []
    for i in range(len(xy)):
        x.append(xy[i][0])
        y.append(xy[i][1])
    return x, y


def separate_with_thresh(data, th):
    after_thresh = [[], []]
    before_thresh = [[], []]
    for i in range(len(data[0])):
        if data[1][i] >= th:
            after_thresh[0].append(data[0][i])
            after_thresh[1].append(data[1][i])
        else:
            before_thresh[0].append(data[0][i])
            before_thresh[1].append(data[1][i])
    return before_thresh, after_thresh


def plot_score(t_x, t_y, d_x, d_y, th, h):
    plt.plot(t_x, t_y, linewidth=0.7, label='Target')
    plt.plot(d_x, d_y, linewidth=0.7, label='Decoy')
    plt.vlines(th, 0, 0.5 * h, linewidth=0.5, label='Threshold')
    plt.legend(loc='upper right')
    plt.savefig('./fdr_plot.png', dpi=500)
    plt.close()


def add_to_labels(data, labels, positive):
    if positive:
        for i in range(len(data[0])):
            labels[data[0][i] - 1] = 1
    else:
        for i in range(len(data[0])):
            labels[data[0][i] - 1] = 0
    return labels


class Label:
    def __init__(self, tsv_addr, output_addr, fdr=0.01, decoy='DeBruijn', log=True):
        self.tsv_addr = tsv_addr
        self.fdr = fdr
        self.decoy = decoy
        self.log = log
        self.output = '{}/labels.csv'.format(output_addr)
        self.scan_num_index = 0
        self.e_value_index = 0
        self.protein_index = 0

    def read(self):
        with open(self.tsv_addr) as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter="\t")

            int_tsv = []
            counter = 0
            for line in tsv_reader:
                int_tsv.append(line)
                if counter == 0:
                    self.scan_num_index = line.index('ScanNum')
                    self.e_value_index = line.index('EValue')
                    self.protein_index = line.index('Protein')
                else:
                    int_tsv[counter][self.scan_num_index] = int(int_tsv[counter][self.scan_num_index])
                counter += 1
            first_row = int_tsv.pop(0)
            if self.log:
                print(len(int_tsv))
            return int_tsv

    def calculate_fdr(self, int_tsv, plot=False):
        sorted_tsv = sorted(int_tsv, key=operator.itemgetter(self.scan_num_index), reverse=False)
        unified_sorted_tsv = [sorted_tsv[0]]

        for i in range(1, len(sorted_tsv)):
            if sorted_tsv[i][self.scan_num_index] == unified_sorted_tsv[-1][self.scan_num_index]:
                if float(sorted_tsv[i][self.e_value_index]) < float(unified_sorted_tsv[-1][self.e_value_index]):
                    unified_sorted_tsv[-1] = sorted_tsv[i]
                elif float(sorted_tsv[i][self.e_value_index]) == float(unified_sorted_tsv[-1][self.e_value_index]):
                    if self.decoy in unified_sorted_tsv[-1][self.e_value_index]\
                            and self.decoy not in sorted_tsv[i][self.e_value_index]:
                        unified_sorted_tsv[-1] = sorted_tsv[i]
            else:
                unified_sorted_tsv.append(sorted_tsv[i])

        decoy, target = [[], []], [[], []]
        for line in unified_sorted_tsv:
            if self.decoy in line[self.protein_index]:
                decoy[0].append(int(line[self.scan_num_index]))
                decoy[1].append(-10 * np.log10(float(line[self.e_value_index])))
            else:
                target[0].append(int(line[self.scan_num_index]))
                target[1].append(-10 * np.log10(float(line[self.e_value_index])))
        if self.log:
            print(len(decoy[0]))
            print(len(target[0]))

        target_density, target_keys = calculate_density(target)
        decoy_density, decoy_keys = calculate_density(decoy)
        target_xy = get_xy(target_density, target_keys)
        decoy_xy = get_xy(decoy_density, decoy_keys)
        target_x, target_y = separate_x_y(target_xy)
        decoy_x, decoy_y = separate_x_y(decoy_xy)

        threshold = min(target[1])
        fdr = 1
        while fdr > self.fdr:
            decoy_after_thresh = separate_with_thresh(decoy, threshold)[1]
            target_after_thresh = separate_with_thresh(target, threshold)[1]
            fp = len(decoy_after_thresh[0])
            tp = len(target_after_thresh[0])
            fdr = fp / (fp + tp)
            threshold += 1

        threshold -= 1
        if self.log:
            print(threshold)
        target_before_thresh, target_after_thresh = separate_with_thresh(target, threshold)
        decoy_before_thresh, decoy_after_thresh = separate_with_thresh(decoy, threshold)
        if self.log:
            print(len(target_after_thresh[0]))
        if plot:
            height = max(target_y)
            plot_score(target_x, target_y, decoy_x, decoy_y, threshold, height)
        return target_before_thresh, target_after_thresh, decoy_before_thresh, decoy_after_thresh

    def write(self, target_before_thresh, target_after_thresh, decoy_before_thresh, decoy_after_thresh):
        last_scan = max([max(target_before_thresh[0]), max(decoy_before_thresh[0]),
                         max(target_after_thresh[0]), max(decoy_after_thresh[0])])

        final_labels = np.zeros((last_scan, 1))
        final_labels = add_to_labels(target_before_thresh, final_labels, True)
        final_labels = add_to_labels(decoy_before_thresh, final_labels, False)
        final_labels = add_to_labels(target_after_thresh, final_labels, True)
        final_labels = add_to_labels(decoy_after_thresh, final_labels, False)

        np.savetxt(self.output, final_labels, fmt='%i')

    def label(self):
        int_tsv = self.read()
        target_before_thresh, target_after_thresh, decoy_before_thresh, decoy_after_thresh = self.calculate_fdr(int_tsv)
        self.write(target_before_thresh, target_after_thresh, decoy_before_thresh, decoy_after_thresh)
