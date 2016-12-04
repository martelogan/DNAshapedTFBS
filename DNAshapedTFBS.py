#!/usr/bin/python2.7
#*-* coding: utf-8 *-*

""" Train and apply PSSM/TFFM/4-bits + DNAshape classifiers. """

import os
PATH = os.path.dirname(os.path.realpath(__file__))
import sys


class InternalArgu:
    def __init__(self, fg_fasta, fg_bed, bg_fasta, bg_bed, output, in_bed, in_fasta):
        self.fg_fasta = fg_fasta
        self.fg_bed = fg_bed
        self.bg_fasta = bg_fasta
        self.bg_bed = bg_bed
        self.output = output
        self.in_bed = in_bed
        self.in_fasta = in_fasta


# Local environment
# TODO: Test if TFFM is installed instead of using local env.
sys.path.append('/Users/Fred/Documents/chipseq/DNAshapedTFBS/TFFM')
from sklearn.externals import joblib
from argparsing import *
from the_constants import BWTOOL, DNASHAPEINTER
from shapes import *
from utilities import *


def find_pssm_hits(pssm, seq_file, isForeground):
    """ Predict hits in sequences using a PSSM. """
    from operator import itemgetter
    import math
    import Bio.SeqIO
    from Bio.Alphabet import generic_dna
    from Bio.Alphabet.IUPAC import IUPACUnambiguousDNA as unambiguousDNA
    import tffm_module
    from hit_module import HIT
    hits = []
    count = 0
    for record in Bio.SeqIO.parse(seq_file, "fasta", generic_dna):
        # see how many records it sees directly reading from FASTA
        count = count + 1
        print(count)
        record.seq.alphabet = unambiguousDNA()
        scores = [(pos, ((score - pssm.min) / (pssm.max - pssm.min)))
                  for pos, score in pssm.search(record.seq, pssm.min) if not
                  math.isnan(score)]
        # pos_i, score_i = max(scores, key=itemgetter(1)) if isForeground else min(scores, key=itemgetter(1))
        pos_i, score_i = max(scores, key=itemgetter(1))
        strand = "+"
        if pos_i < 0:
            strand = "-"
            pos_i = pos_i + len(record.seq)
        hits.append(HIT(record, pos_i + 1, pos_i + pssm.length, strand,
                        score_i))
    return hits


def find_tffm_hits(xml, seq_file, tffm_kind):
    """ Predict hits in sequences using a TFFM. """
    #import sys
    #sys.path.append("/raid6/amathelier/TFFM+DNAshape/bin/TFFM/")
    import tffm_module
    from constants import TFFM_KIND  # TFFM-framework
    if tffm_kind == 'first_order':
        tffm_kind = TFFM_KIND.FIRST_ORDER
    elif tffm_kind == 'detailed':
        tffm_kind = TFFM_KIND.DETAILED
    else:
        sys.exit('The type of TFFM should be "first_order" or "detailed".')
    tffm = tffm_module.tffm_from_xml(xml, tffm_kind)
    return [hit for hit in
            tffm.scan_sequences(seq_file, only_best=True) if hit]


def construct_classifier_input(foreground, background):
    """ Make list of classes for foreground and background. """
    classes = [1.0] * len(foreground) + [0.0] * len(background)
    return foreground + background, classes


def fit_classifier(fg_train_hits, fg_train_shapes, bg_train_hits,
                   bg_train_shapes, extension=0, bool4bits=False):
    """ Fit the classifier to the training data. """
    from sklearn.ensemble import GradientBoostingClassifier
    fg_train = combine_hits_shapes(fg_train_hits, fg_train_shapes, extension,
            bool4bits)
    bg_train = combine_hits_shapes(bg_train_hits, bg_train_shapes, extension,
            bool4bits)
    data, classification = construct_classifier_input(fg_train, bg_train)
    classifier = GradientBoostingClassifier()
    classifier.fit(data, classification)
    return classifier


def make_predictions(clf, tests, hits, thr):
    """ Predict hits from the tests using the classifier. """
    predictions = {'peak_id': [], 'start': [], 'end': [], 'strand': [],
                   'sequence': [], 'proba': []}
    for indx, proba in enumerate(clf.predict_proba(tests)):
        #IIIITTTTTTS FUCKING HERE!!!!!!!!!!
        if proba[1] >= thr:
            hit = hits[indx]
            if hit:
                predictions['peak_id'].append(hit.seq_record.name)
                predictions['start'].append(hit.start)
                predictions['end'].append(hit.end)
                predictions['strand'].append(hit.strand)
                if hit.strand == '-':
                    sequence = ''.join(
                        hit.seq_record.seq[
                            hit.start - 1:hit.end].reverse_complement())
                else:
                    sequence = ''.join(hit.seq_record[hit.start - 1:hit.end])
                predictions['sequence'].append(sequence)
                predictions['proba'].append(proba[1])
    return predictions


def print_predictions(predictions, output):
    """ Print the predictions in the output file. """
    import pandas as pd
    pd_predictions = pd.DataFrame(predictions)
    pd.set_option('display.max_rows', len(pd_predictions))
    with open(output, 'w') as stream:
        stream.write('{0}\n'.format(pd_predictions.to_string(
            index=False, columns=['peak_id', 'start', 'end', 'strand',
                                  'sequence', 'proba'])))


def apply_classifier(hits, argu, bool4bits=False):
    """ Apply the DNAshape based classifier. """
    # Two options, 1) doing sequence by sequence but it means doing a lot of
    # bwtool calls and I/O, 2) doing all sequences as one batch but means that
    # we need to associate back probas to hits. I choose 2) to reduce I/O.

    hits_shapes = get_shapes(hits, argu.fg_bed, argu.first_shape,
            argu.second_shape, argu.extension, argu.scaled)
    classifier = joblib.load(argu.classifier)
    if bool4bits:
        tests = combine_hits_shapes(encode_hits(hits), hits_shapes,
                                    argu.extension, bool4bits)
    else:
        tests = combine_hits_shapes(hits, hits_shapes, argu.extension,
                                    bool4bits)
    # Need to print the results by associating the probas to the hits
    predictions = make_predictions(classifier, tests, hits, argu.threshold)
    print_predictions(predictions, argu.output)


def tffm_apply_classifier(argu):
    """ Apply the TFFM + DNA shape classifier. """
    hits = find_tffm_hits(argu.tffm_file, argu.in_fasta, argu.tffm_kind)
    if hits:
        apply_classifier(hits, argu)
    else:
        with open(argu.output, 'w') as stream:
            stream.write('No hit predicted\n')


def pssm_apply_classifier(argu):
    """ Apply the TFFM + DNA shape classifier. """
    if argu.jasparid:
        pssm = get_jaspar_pssm(argu.jasparid)
    else:
        pssm = get_jaspar_pssm(argu.jasparfile, False)
    hits = find_pssm_hits(pssm, argu.in_fasta)
    if hits:
        apply_classifier(hits, argu)
    else:
        with open(argu.output, 'w') as stream:
            stream.write('No hit predicted\n')


def binary_apply_classifier(argu):
    """ Apply the 4-bits + DNA shape classifier. """
    if argu.jasparid:
        pssm = get_jaspar_pssm(argu.jasparid)
    else:
        pssm = get_jaspar_pssm(argu.jasparfile, False)
    hits = find_pssm_hits(pssm, argu.in_fasta)
    if hits:
        apply_classifier(hits, argu, True)
    else:
        with open(argu.output, 'w') as stream:
            stream.write('No hit predicted\n')


def train_classifier(fg_hits, bg_hits, argu, bool4bits=False):
    """ Train the DNAshape-based classifier. """
    fg_shapes = get_shapes(fg_hits, argu.fg_bed, argu.first_shape,
            argu.second_shape, argu.extension, argu.scaled)
    bg_shapes = get_shapes(bg_hits, argu.bg_bed, argu.first_shape,
            argu.second_shape, argu.extension, argu.scaled)
    if bool4bits:
        classifier = fit_classifier(encode_hits(fg_hits), fg_shapes,
                encode_hits(bg_hits), bg_shapes, argu.extension, bool4bits)
    else:
        classifier = fit_classifier(fg_hits, fg_shapes, bg_hits, bg_shapes,
                argu.extension, bool4bits)
    joblib.dump(classifier, '{0}.pkl'.format(argu.output))


def kFoldClassification(data, labels, classifier, cv):
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import datasets
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    """ Run Kfold classification on a machine learning classifier """

    # y-axis (dependent variable) = sensitivity
    mean_tpr = 0.0
    # x-axis (independent variable) = specificity
    mean_fpr = np.linspace(0, 1, 100)
    # line colors
    colors = cycle(['indigo', 'blue', 'darkorange'])
    # line width
    lw = 2

    #convert to numpy array
    data = np.array(data)
    labels = np.array(labels)

    # Perform and plot ROC curve for each fold
    i = 0
    for (train, test), color in zip(cv.split(data, labels), colors):
        print "\n\ntrain\n\n", train
        print "\n\ntest\n\n", test
        print "\n\nDATA\n\n", data[train]
        print "\n\nLABELS\n\n", labels[train]
        probas_ = classifier.fit(data[train], labels[train]).predict_proba(data[test])
        # Compute ROC curve and area the curve
        print "\n\nPROBAS_\n\n", probas_[:, 1]
        print "\n\nLABELS\n\n", labels[test]
        fpr, tpr, thresholds = roc_curve(labels[test], probas_[:, 1])
        print "\n\nfpr\n\n", fpr
        print "\n\ntpr\n\n", tpr
        print "\n\nthresholds\n\n", thresholds
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        print "\n\nmean_tpr\n\n", mean_tpr
        print "\n\nroc_auc\n\n", roc_auc
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(data, labels)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def tffm_train_classifier(argu):
    """ Train a TFFM + DNA shape classifier. """
    fg_hits = find_tffm_hits(argu.tffm_file, argu.fg_fasta, argu.tffm_kind)
    bg_hits = find_tffm_hits(argu.tffm_file, argu.bg_fasta, argu.tffm_kind)
    train_classifier(fg_hits, bg_hits, argu)


def pssm_train_classifier(argu):
    """ Train a PSSM + DNA shape classifier. """
    if argu.jasparid:
        pssm = get_jaspar_pssm(argu.jasparid)
    else:
        pssm = get_jaspar_pssm(argu.jasparfile, False)
    fg_hits = find_pssm_hits(pssm, argu.fg_fasta)
    bg_hits = find_pssm_hits(pssm, argu.bg_fasta)
    train_classifier(fg_hits, bg_hits, argu)
#SPLIT HERE, CHANGE THE ARGU FOR EACH SO THAT ITS CORRECT FOR NAME, SIZE ETC...

def binary_train_classifier(argu):
    """ Train a 4-bits + DNA shape classifier. """
    if argu.jasparid:
        pssm = get_jaspar_pssm(argu.jasparid)
    else:
        pssm = get_jaspar_pssm(argu.jasparfile, False)
    fg_hits = find_pssm_hits(pssm, argu.fg_fasta)
    bg_hits = find_pssm_hits(pssm, argu.bg_fasta)
    train_classifier(fg_hits, bg_hits, argu, True)


def pssm_trainAndApply_classifier(argu):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    # Build internal args
    # internalArgu = InternalArgu(argu.fg_fasta, argu.fg_bed, argu.bg_fasta, argu.bg_bed, argu.output, argu.fg_bed, argu.fg_fasta)

    # ********************
    # TRAIN CLASSIFIER
    # ********************
    if argu.jasparid:
        pssm = get_jaspar_pssm(argu.jasparid)
    else:
        pssm = get_jaspar_pssm(argu.jasparfile, False)
    fg_hits = find_pssm_hits(pssm, argu.fg_fasta, True)
    bg_hits = find_pssm_hits(pssm, argu.bg_fasta, False)
    fg_shapes = get_shapes(fg_hits, argu.fg_bed, argu.first_shape,
        argu.second_shape, argu.extension, argu.scaled)
    bg_shapes = get_shapes(bg_hits, argu.bg_bed, argu.first_shape,
        argu.second_shape, argu.extension, argu.scaled)
    """ Fit the classifier to the training data. """
    foreground_data = combine_hits_shapes(fg_hits, fg_shapes)
    background_data = combine_hits_shapes(bg_hits, bg_shapes)
    fg_len = len(foreground_data)
    bg_len = len(background_data)
    if(fg_len > bg_len):
        foreground_data = foreground_data[0:bg_len]
    elif(bg_len > fg_len):
        background_data = background_data[0:fg_len]
    data, classification = construct_classifier_input(foreground_data, background_data)

    # Machine learning estimator
    classifier = GradientBoostingClassifier()

    # Cross-validation parameter
    cv = StratifiedKFold(n_splits=2)

    kFoldClassification(data, classification, classifier, cv)

    # *******************
    # APPLY CLASSIFIER - FOREGROUND
    # ********************

    # hits = find_pssm_hits(pssm, internalArgu.in_fasta, True)
    # if hits:
    #     apply_classifier(hits, internalArgu)
    # else:
    #     with open(internalArgu.output, 'w') as stream:
    #         stream.write('No hit predicted\n')

    # ********************
    # APPLY CLASSIFIER - BACKGROUND
    # ********************

    # SET
    # internalArgu.in_fasta = internalArgu.bg_fasta
    # internalArgu.in_bed = internalArgu.bg_bed

    # hits = find_pssm_hits(pssm, internalArgu.in_fasta, False)
    # if hits:
    #     apply_classifier(hits, internalArgu)
    # else:
    #     with open(internalArgu.output, 'w') as stream:
    #         stream.write('No hit predicted\n')

##############################################################################
#                               MAIN
##############################################################################
if __name__ == "__main__":
    arguments = arg_parsing()
    arguments.func(arguments)

