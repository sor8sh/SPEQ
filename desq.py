import os

from classes.mgf_sequencer_1d_out import Sequencer
from classes.labels_out import Label
from classes.model_out import DeSQ
import yaml

config = yaml.safe_load(open('./config.yml'))

input_tsv_label = config['db result file']
input_mgf = config['MS data']
input_dir = os.path.dirname(input_mgf)
out_dir = config['output directory']

if out_dir is None:
    out_dir = '{}/DeSQ_out'.format(input_dir)
    try:
        os.mkdir('{}/DeSQ_out'.format(input_dir))
    except FileExistsError:
        pass
else:
    try:
        os.mkdir('{}/DeSQ_out'.format(out_dir))
    except FileExistsError:
        pass
    out_dir += '/DeSQ_out'

# creating labels
if not config['label is ready']:
    label = Label(input_tsv_label, out_dir, fdr=float(config['fdr']), decoy=config['decoy prefix'], log=config['labels log'])
    label.label()

# creating data
sequencer = Sequencer(input_mgf, out_dir, log=config['sequencer log'])
sequencer.sequence()

# building model
model = DeSQ(out_dir, train_to_test=config['train to test'],
             log=config['model log'], dataset_addr=config['dataset directory'])

results = model.run(config['mode'], batch_n=int(config['batch size']), epoch_n=int(config['number of epochs']),
                    verbose=int(config['verbose']), validation_split=float(config['validation split']),
                    load_weights=config['load weights'], save_weights=config['save weights'],
                    weight_id=config['weight id'])

# calculate AUC
if config['AUC']:
    auc = model.get_auc()
    print(auc)

# plotting ROC curve
if config['ROC plot']:
    model.get_roc_plot()

# calculating confusion matrix
if config['confusion matrix']:
    tp, fp, tn, fn = model.get_confusion_matrix()
    print(tp, fp, tn, fn)

# creating results file
if config['results file']:
    model.get_results()
