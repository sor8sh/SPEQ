from pyteomics import mgf
import pandas as pd
import csv
from classes.Utils import Utils


class Sequencer:
    def __init__(self, data_addr, ms_file, log=True):
        self.labels = f'{data_addr}/labels.csv'
        self.input_mgf = f'{data_addr}/{ms_file}'
        self.output = f'{data_addr}/data.csv'
        self.log = log

    def sequence(self):
        dataset = self.read()
        self.write(dataset)

    def read(self):
        df = pd.read_csv(self.labels, delimiter=',', index_col=0)
        with mgf.read(self.input_mgf) as reader:
            labels = df["Label"].to_dict()
            dataset, total = [], len(reader)
            for i, spectrum in enumerate(reader):
                title = spectrum['params']['title']
                charge = str(spectrum['params']['charge']) if 'charge' in spectrum['params'].keys() else '0+'
                precursor_mz = round(spectrum['params']['pepmass'][0], 2)
                one_d_seq = Utils.get_one_d_seq(spectrum['intensity array'], spectrum['m/z array'])
                dataset.append([title, int(labels[title]), [charge, precursor_mz], one_d_seq])
                Utils.progress(self.log, 'Reading MGF file...', i, total)
        return dataset

    def write(self, dataset):
        with open(self.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Spectrum', 'Label', 'Feature1 (Charge State)', 'Feature2 (Precursor m/z)', 'OSO'])
            total = len(dataset)
            for i, (spectrum, label, features, peaks) in enumerate(dataset):
                row = [spectrum, label, features[0][:-1], features[1], ' '.join(map(str, map(int, peaks)))]
                writer.writerow(row)
                Utils.progress(self.log, 'Generating dataset...', i, total)
