# *-* coding: utf-8 *-*

""" All classification executions 
(train/apply/kfold for each of: PSSM/TFFM/4-bits * DNAshape * DNAflex classifiers). 
"""

# LIBRARY IMPORTS
from sklearn.externals import joblib
from DNAshapedTFBS_argsParsing import *
from DNAshapedTFBS_featureVectors import *
from DNAshapedTFBS_commonUtils import *

# Local environment config
# TODO: Test if TFFM is installed instead of using local env.
PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{0}/TFFM/'.format(PATH))


# CLASSIFICATION HELPER FUNCTIONS


def construct_classifier_input(foreground, background):
    """ Make list of classes for foreground and background. """
    classes = [1.0] * len(foreground) + [0.0] * len(background)
    return foreground + background, classes


def fit_classifier(fg_train_hits, fg_train_shapes, bg_train_hits,
                   bg_train_shapes, extension=0, bool4bits=False):
    """ Fit the classifier to the training data. """
    from sklearn.ensemble import GradientBoostingClassifier
    fg_train = construct_dna_shape_only_feature_vector(fg_train_hits, fg_train_shapes, extension,
                                                       bool4bits)
    bg_train = construct_dna_shape_only_feature_vector(bg_train_hits, bg_train_shapes, extension,
                                                       bool4bits)
    data, classification = construct_classifier_input(fg_train, bg_train)
    classifier = GradientBoostingClassifier()
    classifier.fit(data, classification)
    return classifier


# CLASSIFICATION EXECUTIONS


def apply_classifier(hits, argu, bool4bits=False):
    """ Apply the DNAshape based classifier. """
    # Two options, 1) doing sequence by sequence but it means doing a lot of
    # bwtool calls and I/O, 2) doing all sequences as one batch but means that
    # we need to associate back probas to hits. I choose 2) to reduce I/O.

    hits_shapes = get_motif_dna_shape_vector(hits, argu.fg_bed, argu.first_shape,
                                             argu.second_shape, argu.extension, argu.scaled)
    classifier = joblib.load(argu.classifier)
    if bool4bits:
        tests = construct_dna_shape_only_feature_vector(encode_hits(hits), hits_shapes,
                                                        argu.extension, bool4bits)
    else:
        tests = construct_dna_shape_only_feature_vector(hits, hits_shapes, argu.extension,
                                                        bool4bits)
    # Need to print the results by associating the probas to the hits
    predictions = make_predictions(classifier, tests, hits, argu.threshold)
    output_classifier_predictions(predictions, argu.output)


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
    fg_shapes = get_motif_dna_shape_vector(fg_hits, argu.fg_bed, argu.first_shape,
                                           argu.second_shape, argu.extension, argu.scaled)
    bg_shapes = get_motif_dna_shape_vector(bg_hits, argu.bg_bed, argu.first_shape,
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


def pssm_train_and_apply_classifier(argu):
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
    fg_shapes = get_motif_dna_shape_vector(fg_hits, argu.fg_bed, argu.first_shape,
                                           argu.second_shape, argu.extension, argu.scaled)
    bg_shapes = get_motif_dna_shape_vector(bg_hits, argu.bg_bed, argu.first_shape,
                                           argu.second_shape, argu.extension, argu.scaled)
    # boolean below causes flex to use eval function
    is_eval_f = False
    fg_flex = get_promoter_region_flex_vector(fg_hits, is_eval_f)
    bg_flex = get_promoter_region_flex_vector(bg_hits, is_eval_f)
    foreground_data = construct_dna_shape_and_flexibility_feature_vector(fg_hits, fg_shapes, fg_flex)
    background_data = construct_dna_shape_and_flexibility_feature_vector(bg_hits, bg_shapes, bg_flex)
    fg_len = len(foreground_data)
    bg_len = len(background_data)
    if fg_len > bg_len:
        foreground_data = foreground_data[0:bg_len]
    elif bg_len > fg_len:
        background_data = background_data[0:fg_len]
    data, classification = construct_classifier_input(foreground_data, background_data)

    # Machine learning estimator
    classifier = GradientBoostingClassifier()

    # Cross-validation parameter
    cv = StratifiedKFold(n_splits=5)

    # K_fold execution
    pssm_k_fold_classification(data, classification, classifier, cv, argu, pssm.length, is_eval_f)

    # *******************************
    # APPLY CLASSIFIER - FOREGROUND
    # *******************************

    # hits = find_pssm_hits(pssm, internalArgu.in_fasta, True)
    # if hits:
    #     apply_classifier(hits, internalArgu)
    # else:
    #     with open(internalArgu.output, 'w') as stream:
    #         stream.write('No hit predicted\n')

    # *******************************
    # APPLY CLASSIFIER - BACKGROUND
    # *******************************

    # internalArgu.in_fasta = internalArgu.bg_fasta
    # internalArgu.in_bed = internalArgu.bg_bed

    # hits = find_pssm_hits(pssm, internalArgu.in_fasta, False)
    # if hits:
    #     apply_classifier(hits, internalArgu)
    # else:
    #     with open(internalArgu.output, 'w') as stream:
    #         stream.write('No hit predicted\n')


def pssm_k_fold_classification(data, labels, classifier, cv, argu, motif_length, is_eval_f):
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle
    import csv

    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    """ Run Kfold classification on a machine learning classifier """

    # CONVERT TO NUMPY ARRAYS PRIOR TO EXECUTION
    data = np.array(data)
    labels = np.array(labels)

    #############################################
    # # ******** K-fold ROC + PRC's *************
    # # ********   BEGIN-SECTION    *************
    # ###########################################

    ################
    # GRAPH PREP
    ################
    # line colors
    colors = cycle(['indigo', 'blue', 'darkorange', 'yellow', 'green'])
    # line width
    lw = 1

    ################
    # PRC VALUES
    ################
    reversed_mean_precision = 0.0
    mean_recall = np.linspace(0, 1, 100)

    ################
    # ROC VALUES
    ################
    # y-axis (dependent variable) = sensitivity
    mean_tpr = 0.0
    # x-axis (independent variable) = specificity
    mean_fpr = np.linspace(0, 1, 100)

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
    prc.axhline(y=0.5, xmin=0.05, xmax=1, c="black", linewidth=lw, linestyle='--', label='Luck')
    prc.set_xlim([-0.05, 1.05])
    prc.set_ylim([-0.05, 1.05])
    prc.set_xlabel('Recall')
    prc.set_ylabel('Precision')
    prc.set_title('Precision Recall Curve For Protein: ' + argu.output)
    prc.legend(loc="lower right", prop={'size': 12})

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
    roc.legend(loc="lower right", prop={'size': 12})
    fig.savefig(argu.output + "_roc_prc.png", bbox_inches='tight')

    # SAVE SOME VALUES TO CSV
    # protein, avg AUPRC, avg AUROC
    fields = [argu.output, str(mean_auc_pr), str(mean_auc)]
    with open(r'AUPRC_AUROC.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    #############################################
    # # ******** K-fold ROC + PRC's *************
    # # ********   END-SECTION    *************
    # ###########################################

    #############################################
    # # ******** FEATURE IMPORTANCE *************
    # # ********   BEGIN-SECTION    *************
    # ###########################################

    classifier.fit(data, labels)
    importances = classifier.feature_importances_
    # for n_estimators, loss_k = classifier.estimators_
    # std = np.std([tree.feature_importances_ for tree in n_estimators],
    #             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    shape_feature_names = ['HelT', 'ProT', 'MGW', 'Roll', 'HelT2', 'MGW2', 'Roll2']
    print "\n\nOur shape features:", shape_feature_names
    print "\n\nOur motif length:", motif_length

    feature_names = []
    for shapeName in shape_feature_names:
        feature_names += [shapeName] * motif_length

    if is_eval_f:  # we used the eval function feature
        flexibility_eval_function_str = ['Flex_Eval_Function']
        feature_names += flexibility_eval_function_str
    else:  # we used the trinucleotide counts directly
        tri_nuc_classes = ['AAT', 'AAA', 'CCA', 'AAC', 'ACT', 'CCG', 'ATC', 'AAG', 'CGC', 'AGG', 'GAA', 'ACG', 'ACC',
                           'GAC', 'CCC', 'ACA', 'CGA', 'GGA', 'CAA', 'AGC', 'GTA', 'AGA', 'CTC', 'CAC', 'TAA', 'GCA',
                           'CTA', 'GCC', 'ATG', 'CAG', 'ATA', 'TCA']
        feature_names += tri_nuc_classes

    # NOTE: data.shape[1] below is a call to numpy for the dimension m of our n x m data matrix
    for row_number in range(data.shape[1]):
        # SAVE SOME VALUES TO CSV
        # protein, featureName, importance
        fields = [argu.output, feature_names[indices[row_number] - 1], importances[indices[row_number]]]
        with open(r'FEATURE_IMPORTANCES.csv', 'a') as feature_importances_file:
            writer = csv.writer(feature_importances_file)
            writer.writerow(fields)

            #############################################
            # # ******** FEATURE IMPORTANCE *************
            # # ********   END-SECTION    *************
            # ###########################################


##############################################################################
#                               MAIN
##############################################################################


if __name__ == "__main__":
    arguments = arg_parsing()
    arguments.func(arguments)
