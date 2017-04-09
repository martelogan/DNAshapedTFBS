# GLOBAL IMPORTS
import os
import sys

# LIBRARY IMPORTS
import numpy as np

from itertools import cycle
from matplotlib import pyplot as plt
from sklearn.metrics import auc

from DNAshapedTFBS_constants import *
from DNAshapedTFBS_common_utils import feature_vector_type_to_string, seq_feature_type_to_string
from DNAshapedTFBS_constants import CUMULATIVE_AUPRC_AUROC_PATH

PATH = os.path.dirname(os.path.realpath(__file__))

# GLOBAL PLOTTING PARAMETERS
# basic plot settings
colors = cycle(['indigo', 'blue', 'darkorange', 'yellow', 'green'])
lw = 1
# prc params
reversed_mean_precision = 0.0
mean_recall = np.linspace(0, 1, 100)
# roc params
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
# prc_roc figure params
fig = plt.figure()
prc = fig.add_subplot(211)
roc = fig.add_subplot(212)


def init_basic_plot_settings(user_colors, user_lw):
    global colors
    global lw
    colors = user_colors
    lw = user_lw


def init_prc_params(user_reversed_mean_precision, user_mean_recall):
    global reversed_mean_precision
    global mean_recall
    reversed_mean_precision = user_reversed_mean_precision
    mean_recall = user_mean_recall


def init_roc_params(user_mean_tpr, user_mean_fpr):
    global mean_tpr
    global mean_fpr
    mean_tpr = user_mean_tpr
    mean_fpr = user_mean_fpr


def init_prc_and_roc_figure(user_fig, user_prc, user_roc):
    global fig
    global prc
    global roc
    fig = user_fig
    prc = user_prc
    roc = user_roc


def add_single_fold_prc_to_figure(precision, recall, color, fold_number):
    import numpy as np
    from scipy import interp

    from sklearn.metrics import auc

    global reversed_mean_precision

    reversed_recall = np.fliplr([recall])[0]
    reversed_precision = np.fliplr([precision])[0]
    reversed_mean_precision += interp(mean_recall, reversed_recall, reversed_precision)
    reversed_mean_precision[0] = 0.0

    prc_auc = auc(recall, precision)
    prc.plot(recall, precision, lw=lw, color=color,
             label='PRC fold %d (area = %0.6f)' % (fold_number, prc_auc))


def add_single_fold_roc_to_figure(fpr, tpr, color, fold_number):
    from scipy import interp

    from sklearn.metrics import auc

    global mean_tpr

    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0

    roc_auc = auc(fpr, tpr)
    roc.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.6f)' % (fold_number, roc_auc))


def aggregate_k_prc_folds(argu, n_splits):
    global reversed_mean_precision
    global mean_recall
    reversed_mean_precision /= n_splits
    reversed_mean_precision[0] = 1
    mean_auprc = auc(mean_recall, reversed_mean_precision)
    prc.plot(np.fliplr([mean_recall])[0], np.fliplr([reversed_mean_precision])[0], color='g', linestyle='--',
             label='Mean PRC (area = %0.6f)' % mean_auprc, lw=lw)
    prc.axhline(y=0.5, xmin=0.05, xmax=1, c="black", linewidth=lw, linestyle='--', label='Luck')
    prc.set_xlim([-0.05, 1.05])
    prc.set_ylim([-0.05, 1.05])
    prc.set_xlabel('Recall')
    prc.set_ylabel('Precision')
    prc.set_title('Precision Recall Curve For Protein: ' + argu.output)
    prc.legend(loc="lower right", prop={'size': 12})
    return mean_auprc


def aggregate_k_roc_folds(argu, n_splits):
    global mean_tpr
    global mean_fpr

    roc.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')
    mean_tpr /= n_splits
    mean_tpr[-1] = 1.0
    mean_auroc = auc(mean_fpr, mean_tpr)
    roc.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.6f)' % mean_auroc, lw=lw)

    roc.set_xlim([-0.05, 1.05])
    roc.set_ylim([-0.05, 1.05])
    roc.set_xlabel('False Positive Rate')
    roc.set_ylabel('True Positive Rate')
    roc.set_title('Receiver operating characteristic For Protein: ' + argu.output)
    roc.legend(loc="lower right", prop={'size': 12})
    return mean_auroc


def output_k_fold_prc_roc_results(argu, feature_vector_type, seq_feature_type, mean_auprc, mean_auroc):
    import csv
    fig.savefig(argu.output + "_prc_roc.png", bbox_inches='tight')

    # WRITE AVERAGE VALUES FOR THIS EXPERIMENT TO CSV
    if not os.path.isfile(CUMULATIVE_AUPRC_AUROC_PATH):
        with open(r'' + CUMULATIVE_AUPRC_AUROC_PATH, 'w') as f:
            writer = csv.writer(f)
            headers = ['Experiment_Name', 'Feature_Vector_Type', 'Sequence_Feature_Type',
                       'Background_Type', 'Protein', 'AUPRC', 'AUROC']
            writer.writerow(headers)
    feature_vector_type_str = feature_vector_type_to_string(feature_vector_type)
    seq_feature_type_str = seq_feature_type_to_string(seq_feature_type)
    try:
        exp_name = argu.exp_name
    except AttributeError:
        exp_name = 'Unnamed'
    try:
        background_type_str = argu.background_type
    except AttributeError:
        background_type_str = 'N/A'
    titles = [exp_name, feature_vector_type_str, seq_feature_type_str, background_type_str]
    try:
        protein_name = argu.protein
    except AttributeError:
        protein_name = argu.output
    fields = titles + [protein_name, str(mean_auprc), str(mean_auroc)]
    with open(r'' + CUMULATIVE_AUPRC_AUROC_PATH, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


