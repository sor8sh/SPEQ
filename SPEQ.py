import os
import yaml
from classes.Utils import Utils
from classes.mgf_sequencer import Sequencer
from classes.labels import Label
from classes.model import SPEQ

config = yaml.safe_load(open('./config.yml'))
data_dir = os.path.abspath(config['data directory'] or os.getcwd())
out_dir = os.path.join(data_dir, 'out')
Utils.create_directory(data_dir)
Utils.create_directory(out_dir)
if config['save weights']: os.makedirs(f'{out_dir}/weights', exist_ok=True)

# creating labels
if not config['labels.csv']:
    label = Label(
        tsv_addr=config['db result file'],
        output_addr=data_dir,
        fdr=float(config['fdr']),
        decoy=config['decoy prefix'],
        log=config['labels log'])
    label.label()

# creating data
sequencer = Sequencer(
    data_addr=data_dir,
    ms_file=config['MS file'],
    log=config['sequencer log'])
sequencer.sequence()

# building model
speq = SPEQ(
    data_addr=data_dir,
    output_addr=out_dir,
    train_to_test=config['train to test'],
    log=config['model log'],
    dataset_addr=config['dataset directory'])

speq.load_data()
model = speq.model(
    load_w=config['load weights'],
    batch=int(config['batch size']),
    epoch=int(config['number of epochs']),
    verbose=int(config['verbose']),
    validation_split=float(config['validation split']),
    save_w=config['save weights'],
    w_id=config['weight id'])
speq.test(model)

# calculate AUC
if config['AUC']: print(f'AUC: {speq.get_auc()}')

# plotting ROC curve
if config['ROC plot']: speq.get_roc_plot()

# calculating confusion matrix
if config['confusion matrix']:
    tn, fp, fn, tp = speq.get_confusion_matrix()
    print(f'True Positives: {tp} | False Positives: {fp}')
    print(f'True Negatives: {tn} | False Negatives: {fn}')

# creating results file
if config['results file']: speq.get_results()
