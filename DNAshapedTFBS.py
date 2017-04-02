#!/usr/bin/python2.7
#*-* coding: utf-8 *-*

""" Train and apply PSSM/TFFM/4-bits + DNAshape classifiers. """

import os
PATH = os.path.dirname(os.path.realpath(__file__))
import sys
from Bio import trie
from Bio import triefind

# Local environment
# TODO: Test if TFFM is installed instead of using local env.
sys.path.append('{0}/TFFM/'.format(PATH))
from sklearn.externals import joblib
from argparsing import *
from the_constants import BWTOOL, DNASHAPEINTER
from shapes import *
from utilities import *


# HELPER FUNCTIONS FOR 561-proj


def seq_splice(seq, w_start, w_end, ext_size):
    """  returns start and end index in spliced sequence """
    seq_length = len(seq)
    pos_start, pos_end = 0, 0
    if(w_start >= ext_size):
        if(w_end < (seq_length - ext_size)):  # (w_start >= L) and (w_end <= len(S) - L)
            pos_start = w_start - ext_size
            pos_end = w_end + ext_size
        else:  # (w_start >= L) and (w_end > len(S) - L)
            # (w_start) - (ext_size + (ext_size - (*))
            # = w_start - (2 * ext_size) - (*)
            pos_start = w_start - (2 * ext_size) - (seq_length - w_end - 1)
            pos_end = seq_length - 1
    else:  # (w_start < L) and (w_end < len(S) - L)
        pos_start = 0
        pos_end = w_end + (2 * ext_size) - w_start
    if (pos_start < 0):
        pos_start = 0
    if (pos_end > seq_length - 1):
        pos_end = seq_length - 1
    return pos_start, pos_end


def extended_flex_evals(hits, isEval_f):
    """ evaluate promoter region of hits """
    from itertools import product
    import Bio.SeqIO
    from Bio.Seq import Seq
    from Bio.Alphabet import generic_dna
    from Bio.Alphabet.IUPAC import IUPACUnambiguousDNA as unambiguousDNA
    from Bio.Alphabet.IUPAC import IUPACUnambiguousDNA as unambiguousDNA
    # Build trie structure to evaluate promoter regions
    tri_nuc_classes = ['AAT', 'AAA', 'CCA', 'AAC', 'ACT', 'CCG', 'ATC', 'AAG', 'CGC', 'AGG', 'GAA', 'ACG', 'ACC', 'GAC', 'CCC', 'ACA', 'CGA', 'GGA', 'CAA', 'AGC', 'GTA', 'AGA', 'CTC', 'CAC', 'TAA', 'GCA', 'CTA', 'GCC', 'ATG', 'CAG', 'ATA', 'TCA']
    if (isEval_f):  # eval function trie
        tr_eval = trie.trie()
        pvals = [0.755783741,0.760332075,0.781922225,0.814647316,0.832768156,0.872842632,0.895834135,0.922193691,0.925889854,0.944594069,0.963676135,0.96753856,0.968506582,0.987084135,0.988071713,0.994017964,0.997004496,1.013084867,1.010050167,1.017145322,1.025315121,1.027367803,1.031485504,1.040810774,1.070365308,1.078962574,1.094174284,1.112934254,1.14339282,1.191246217,1.199614194,1.214096283]
        for i in xrange(0, len(tri_nuc_classes)):
            pval = pvals[i]
            word = tri_nuc_classes[i]
            word_seq_record = Seq(word, generic_dna)
            compl_word = str(word_seq_record.reverse_complement())
            tr_eval[word] = pval
            tr_eval[compl_word] = pval
    else:  # counts trie
        # Enumerate all trinucleotide keys
        alphabet = unambiguousDNA()
        trinucleotides = [''.join(i) for i in product(alphabet.letters, repeat = 3)]
    # Iteratively evaluate promoter regions of hits
    flex_evals = []
    for hit in hits:
        if hit:
            hit_seq = hit.seq_record.seq
            # print "Sequence length:", len(hit_seq)
            # print "(hit_start, hit_end) =", (hit.start, hit.end)
            ext_start, ext_end = seq_splice(hit_seq, hit.start, hit.end, 50)
            # print "Hit_seq:", hit_seq, "(start, end) =", (ext_start, ext_end)
            ext_seq = str(hit_seq[ext_start:ext_end + 1].upper())
            # print "Ext_seq:", ext_seq
            if isEval_f: # using eval function
                eval_result = 0.0
                for word in triefind.find(ext_seq, tr_eval):
                    eval_result += tr_eval[word[0]]
                print eval_result
                flex_evals.append([eval_result])
            else: # using counts trie
                counts = []
                tr_count = trie.trie()
                for key in trinucleotides:
                    tr_count[key] = 0
                for word in triefind.find(ext_seq, tr_count):
                    tr_count[word[0]] += 1
                for i in xrange(0, len(tri_nuc_classes)):
                    word = tri_nuc_classes[i]
                    word_seq_record = Seq(word, generic_dna)
                    compl_word = str(word_seq_record.reverse_complement())
                    # print "Sequence:", ext_seq
                    # print "Word:", word, ", count:", tr_count[word]
                    # print "Complement:", compl_word, ", count: ", tr_count[compl_word]
                    # Here?
                    count = tr_count[word] + tr_count[compl_word]
                    # print "Count:", count
                    counts.append(count)
                # print "Counts:", counts
                flex_evals.append(counts)
    return flex_evals



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
    # count = 0
    for record in Bio.SeqIO.parse(seq_file, "fasta", generic_dna):
        # see how many records it sees directly reading from FASTA
        # count = count + 1
        # print(count)
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

def kFoldClassification(data, labels, classifier, cv, argu):
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle
    import csv   

    from sklearn import datasets
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from sklearn.model_selection import StratifiedKFold
    """ Run Kfold classification on a machine learning classifier """

    #######
    # GRAPH THINGS
    #######
    # line colors
    colors = cycle(['indigo', 'blue', 'darkorange', 'yellow', 'green'])
    # line width
    lw = 1

    #######
    # PRC VALUES
    #######
    reversed_mean_precision = 0.0
    mean_recall = np.linspace(0, 1, 100)

    #######
    # ROC VALUES
    #######
    # y-axis (dependent variable) = sensitivity
    mean_tpr = 0.0
    # x-axis (independent variable) = specificity
    mean_fpr = np.linspace(0, 1, 100)

    # convert to numpy array
    data = np.array(data)
    labels = np.array(labels)

    # Initialize the two plots
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)
    fig.set_size_inches(10.5, 16.5)
    prc = fig.add_subplot(211)
    roc = fig.add_subplot(212)

    # Perform and plot ROC, PRC curve for each fold
    i = 0

    for (train, test), color in zip(cv.split(data, labels), colors): 

        probas_ = classifier.fit(data[train], labels[train]).predict_proba(data[test])

        ################
        # PRC SECTION
        ################
        precision, recall, thresholds = precision_recall_curve(labels[test], probas_[:, 1])
        reversed_recall = np.fliplr([recall])[0]
        reversed_precision = np.fliplr([precision])[0]
        reversed_mean_precision += interp(mean_recall, reversed_recall, reversed_precision)
        reversed_mean_precision[0] = 0.0

        roc_auc = auc(recall, precision)
        prc.plot(recall, precision, lw=lw, color=color,
                 label='PRC fold %d (area = %0.6f)' % (i, roc_auc))

        ################
        # ROC SECTION
        ################
        fpr, tpr, thresholds = roc_curve(labels[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        roc_auc = auc(fpr, tpr)
        roc.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.6f)' % (i, roc_auc))

        i += 1

    ################
    # PRC SECTION
    ################
    reversed_mean_precision /= cv.get_n_splits(data, labels)
    reversed_mean_precision[0] = 1
    mean_auc_pr = auc(mean_recall, reversed_mean_precision)
    prc.plot(np.fliplr([mean_recall])[0], np.fliplr([reversed_mean_precision])[0], color='g', linestyle='--',
             label='Mean PRC (area = %0.6f)' % mean_auc_pr, lw=lw)
    prc.axhline(y=0.5,xmin=0.05,xmax=1,c="black",linewidth=lw, linestyle='--', label='Luck')
    prc.set_xlim([-0.05, 1.05])
    prc.set_ylim([-0.05, 1.05])
    prc.set_xlabel('Recall')
    prc.set_ylabel('Precision')
    prc.set_title('Precision Recall Curve For Protein: ' + argu.output)
    prc.legend(loc="lower right", prop={'size':12})

    # SAVE SOME VALUES TO CSV

    ################
    # ROC SECTION
    ################
    roc.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(data, labels)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    roc.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.6f)' % mean_auc, lw=lw)

    roc.set_xlim([-0.05, 1.05])
    roc.set_ylim([-0.05, 1.05])
    roc.set_xlabel('False Positive Rate')
    roc.set_ylabel('True Positive Rate')
    roc.set_title('Receiver operating characteristic For Protein: ' + argu.output)
    roc.legend(loc="lower right", prop={'size':12})
    fig.savefig(argu.output+"_roc_prc.png", bbox_inches='tight')

    # SAVE SOME VALUES TO CSV
    #protein, avg AUPRC, avg AUROC
    fields=[argu.output,str(mean_auc_pr),str(mean_auc)]
    with open(r'AUPRC_AUROC.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)



def construct_classifier_input(foreground, background):
    """ Make list of classes for foreground and background. """
    classes = [1.0] * len(foreground) + [0.0] * len(background)
    return foreground + background, classes

########################################################
################### UNUSED CODE ########################
########################################################

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
    # boolean below causes flex to use eval function
    fg_flex = extended_flex_evals(fg_hits, False)
    bg_flex = extended_flex_evals(bg_hits, False)
    foreground_data = construct_HitShapeFlex_vector(fg_hits, fg_shapes, fg_flex)
    background_data = construct_HitShapeFlex_vector(bg_hits, bg_shapes, bg_flex)
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
    cv = StratifiedKFold(n_splits=5)

    kFoldClassification(data, classification, classifier, cv, argu)

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

