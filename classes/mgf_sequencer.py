from pyteomics import mgf
import numpy as np


def scale(i_list):
    maximum = max(i_list)
    o_list = []
    for i in range(len(i_list)):
        o_list.append(100 * i_list[i] / maximum)
    return o_list


def combine(mz_list, i_list):
    real_peaks = []
    for i in range(len(i_list)):
        # if i_list[i] > 0:
        real_peaks.append((round(mz_list[i], 2), round(i_list[i], 2)))
    return real_peaks


def cut_off(peak, thresh):
    new_peak = []
    for i in range(len(peak)):
        if peak[i][1] > thresh:
            new_peak.append(peak[i])
    return new_peak


def get_one_d_seq(peaks):
    min_peak = int(peaks[0][0])
    max_peak = int(peaks[-1][0])
    seq = [0] * (max_peak - min_peak + 1)
    for i in range(len(peaks)):
        peak = int((peaks[i][0] + 0.4) / 1.0005079)
        if peaks[i][1] > seq[peak - min_peak]:
            seq[peak - min_peak] = peaks[i][1]
    return seq


def get_sig_int(peaks, thresh):
    l = []
    for i in range(len(peaks)):
        if peaks[i] > thresh:
            l.append(peaks[i])
    return l


class Sequencer:
    def __init__(self, i_addr, output_addr, log=True):
        self.input_mgf = i_addr
        self.log = log
        self.labels = '{}/labels.csv'.format(output_addr)
        self.output = '{}/data.csv'.format(output_addr)

    def read(self):
        with mgf.read(self.input_mgf) as reader:
            labels = np.loadtxt(self.labels)
            counter = 0
            dataset = []
            for spectrum in reader:
                # 1D sequence
                try:
                    scan = int(spectrum['params']['title'].split('scan=')[1][:-1])
                except IndexError:
                    scan = -1
                mz = spectrum['m/z array']
                intensity = spectrum['intensity array']
                scaled_int = scale(intensity)
                peaks = combine(mz, scaled_int)
                co_peaks = cut_off(peaks, thresh=1)
                one_d_seq = get_one_d_seq(co_peaks)

                # Features
                if 'charge' in spectrum['params'].keys():
                    charge = str(spectrum['params']['charge'])
                else:
                    charge = '0+'
                precursor_mz = round(spectrum['params']['pepmass'][0], 2)
                features = [charge, precursor_mz]

                dataset.append([scan, labels[scan - 1], features, one_d_seq])
                if self.log:
                    if not counter % 1000:
                        print(counter)
                counter += 1
            return dataset

    def write(self, dataset):
        with open(self.output, 'w') as f:
            f.write('Scan, Label, Feature1 (Charge State), Feature2 (Precursor m/z), OSO\n')
            total = len(dataset)
            for i in range(total):
                scan = dataset[i][0]
                f.write('{},'.format(scan))

                label = dataset[i][1]
                f.write('{},'.format(label))

                features = dataset[i][2]
                f.write('{},{},'.format(features[0][:-1], features[1]))

                peaks = dataset[i][3]
                for j in range(len(peaks)):
                    f.write('{} '.format(int(peaks[j])))
                f.write('\n')

                if self.log:
                    if not i % 1000:
                        print('{:.2f}% ({})'.format(100 * (i / total), i))

    def sequence(self):
        dataset = self.read()
        self.write(dataset)
