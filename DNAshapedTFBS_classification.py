# *-* coding: utf-8 *-*

""" All classification executions 
(train/apply/kfold for each of: PSSM/TFFM/4-bits * DNAshape * DNAflex classifiers). 
"""

# LIBRARY IMPORTS
from sklearn.externals import joblib
from DNAshapedTFBS_argsParsing import *
from DNAshapedTFBS_featureVectors import *
from DNAshapedTFBS_commonUtils import *
from DNAshapedTFBS_constants import *

# Local environment config
# TODO: Test if TFFM is installed instead of using local env.
PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{0}/TFFM/'.format(PATH))

# TODO: UPDATE TO REFLECT CURRENTLY ACTIVE BIGWIGS (MISSING MGW2 RIGHT NOW)
shape_feature_names = ['HelT', 'ProT', 'MGW', 'Roll', 'HelT2', 'MGW2', 'Roll2']


# CLASSIFICATION HELPER FUNCTIONS


def construct_classifier_input(foreground, background):
    """ Make list of classes for foreground and background. """
    classes = [1.0] * len(foreground) + [0.0] * len(background)
    return foreground + background, classes


def fit_classifier(argu, fg_training_motif_hits, fg_training_dna_shapes_matrix, bg_training_motif_hits,
                   bg_training_dna_shapes_matrix, feature_vector_type, seq_feature_type):
    """ Fit the classifier to the training data. """
    from sklearn.ensemble import GradientBoostingClassifier
    fg_train = get_feature_vector(argu, feature_vector_type, seq_feature_type,
                                  fg_training_motif_hits, fg_training_dna_shapes_matrix)
    bg_train = get_feature_vector(argu, feature_vector_type, seq_feature_type,
                                  bg_training_motif_hits, bg_training_dna_shapes_matrix)
    data, labels = construct_classifier_input(fg_train, bg_train)
    classifier = GradientBoostingClassifier()
    classifier.fit(data, labels)
    return classifier


# CLASSIFICATION EXECUTIONS

# ********************
# TRAINING FUNCTIONS
# ********************

def train_classifier(argu, fg_hits, bg_hits, feature_vector_type, seq_feature_type):
    """ Train the DNAshape-based classifier and dump the python object to a temporary file"""
    fg_dna_shapes_matrix = None
    bg_dna_shapes_matrix = None
    if feature_vector_type in DNA_SHAPE_FEATURE_TYPE_CONSTANTS: # Include DNA Shape features
        fg_dna_shapes_matrix = get_motif_dna_shapes_vector(fg_hits, argu.fg_bed, argu.shape_first_order,
                                                argu.shape_second_order, argu.extension, argu.scaled)
        bg_dna_shapes_matrix = get_motif_dna_shapes_vector(bg_hits, argu.bg_bed, argu.shape_first_order,
                                                argu.shape_second_order, argu.extension, argu.scaled)
    # NOTE: DNA Shape Matrices might be null below if our classifier does not use them
    if seq_feature_type == BINARY_ENCODING_TYPE_CONSTANT:  # Binary encoding
        classifier = fit_classifier(argu, encode_hits(fg_hits), fg_dna_shapes_matrix, encode_hits(bg_hits),
                                    bg_dna_shapes_matrix, feature_vector_type, seq_feature_type)
    else:  # some other feature vector type
        classifier = fit_classifier(argu, fg_hits, fg_dna_shapes_matrix, bg_hits,
                                    bg_dna_shapes_matrix, feature_vector_type, seq_feature_type)
    # Persist the python classifier object in a temporary file
    joblib.dump(classifier, '{0}.pkl'.format(argu.output))


def dna_shape_and_tffm_train_classifier(argu):
    """ Train a TFFM + DNA shape classifier. """
    fg_hits = find_tffm_hits(argu.tffm_file, argu.fg_fasta, argu.tffm_kind)
    bg_hits = find_tffm_hits(argu.tffm_file, argu.bg_fasta, argu.tffm_kind)
    train_classifier(argu, fg_hits, bg_hits, SEQ_AND_DNA_SHAPE_TYPE_CONSTANT, TFFM_SCORE_TYPE_CONSTANT)


def dna_shape_and_pssm_train_classifier(argu):
    """ Train a PSSM + DNA shape classifier. """
    if argu.jasparid:
        pssm = get_jaspar_pssm(argu.jasparid)
    else:
        pssm = get_jaspar_pssm(argu.jasparfile, False)
    fg_hits = find_pssm_hits(pssm, argu.fg_fasta, True)
    bg_hits = find_pssm_hits(pssm, argu.bg_fasta, False)
    train_classifier(argu, fg_hits, bg_hits, SEQ_AND_DNA_SHAPE_TYPE_CONSTANT, PSSM_SCORE_TYPE_CONSTANT)


def dna_shape_and_binary_train_classifier(argu):
    """ Train a 4-bits + DNA shape classifier. """
    if argu.jasparid:
        pssm = get_jaspar_pssm(argu.jasparid)
    else:
        pssm = get_jaspar_pssm(argu.jasparfile, False)
    fg_hits = find_pssm_hits(pssm, argu.fg_fasta, True)
    bg_hits = find_pssm_hits(pssm, argu.bg_fasta, False)
    train_classifier(argu, fg_hits, bg_hits, SEQ_AND_DNA_SHAPE_TYPE_CONSTANT, BINARY_ENCODING_TYPE_CONSTANT)


def custom_train_classifier(argu):
    """ Adapt classifier training to input args. """
    fg_hits = get_motif_hits(argu, argu.fg_fasta, True)
    bg_hits = get_motif_hits(argu, argu.bg_fasta, False)
    train_classifier(argu, fg_hits, bg_hits, argu.feature_vector_type, argu.seq_feature)


# ********************
# PREDICTION FUNCTIONS
# ********************


def apply_classifier(argu, motif_hits, feature_vector_type, seq_feature_type):
    """ Apply the DNAshape based classifier. """
    # Two options: 1) doing sequence by sequence but it means doing a lot of
    # bwtool calls and I/O, 2) doing all sequences as one batch but means that
    # we need to associate back probas to hits. I chose 2) to reduce I/O. - by Mathelier

    dna_shapes_matrix = None
    if feature_vector_type in DNA_SHAPE_FEATURE_TYPE_CONSTANTS: # Include DNA Shape features
        dna_shapes_matrix = get_motif_dna_shapes_vector(motif_hits, argu.in_bed, argu.shape_first_order,
                                              argu.shape_second_order, argu.extension, argu.scaled)
    # RELOAD OUR CLASSIFIER
    classifier = joblib.load(argu.classifier)
    # NOTE: DNA Shape Matrix might be null below if our classifier does not use it
    if seq_feature_type == BINARY_ENCODING_TYPE_CONSTANT:  # Binary encoding
        test_data_feature_vectors = get_feature_vector(argu, feature_vector_type, seq_feature_type,
                                                       encode_hits(motif_hits), dna_shapes_matrix)
    else:
        test_data_feature_vectors = get_feature_vector(argu, feature_vector_type, seq_feature_type,
                                                       motif_hits, dna_shapes_matrix)
    # To output results, we must first associate our probas to the motif hits
    predictions = make_predictions(classifier, test_data_feature_vectors, motif_hits, argu.threshold)
    output_classifier_predictions(predictions, argu.output)


def dna_shape_and_pssm_apply_classifier(argu):
    """ Apply the TFFM + DNA shape classifier. """
    if argu.jasparid:
        pssm = get_jaspar_pssm(argu.jasparid)
    else:
        pssm = get_jaspar_pssm(argu.jasparfile, False)
    # To be conservative: treat our test data as foreground and consider the best potential motifs
    motif_hits = find_pssm_hits(pssm, argu.in_fasta, True)
    if motif_hits:
        apply_classifier(argu, motif_hits, SEQ_AND_DNA_SHAPE_TYPE_CONSTANT, PSSM_SCORE_TYPE_CONSTANT)
    else:
        with open(argu.output, 'w') as stream:
            stream.write('No motif hits predicted\n')


def dna_shape_and_tffm_apply_classifier(argu):
    """ Apply the TFFM + DNA shape classifier. """
    motif_hits = find_tffm_hits(argu.tffm_file, argu.in_fasta, argu.tffm_kind)
    if motif_hits:
        apply_classifier(argu, motif_hits, SEQ_AND_DNA_SHAPE_TYPE_CONSTANT, TFFM_SCORE_TYPE_CONSTANT)
    else:
        with open(argu.output, 'w') as stream:
            stream.write('No motif hits predicted\n')


def dna_shape_and_binary_apply_classifier(argu):
    """ Apply the 4-bits + DNA shape classifier. """
    if argu.jasparid:
        pssm = get_jaspar_pssm(argu.jasparid)
    else:
        pssm = get_jaspar_pssm(argu.jasparfile, False)
        # To be conservative: treat our test data as foreground and consider the best potential motifs
    motif_hits = find_pssm_hits(pssm, argu.in_fasta, True)
    if motif_hits:
        apply_classifier(argu, motif_hits, SEQ_AND_DNA_SHAPE_TYPE_CONSTANT, BINARY_ENCODING_TYPE_CONSTANT)
    else:
        with open(argu.output, 'w') as stream:
            stream.write('No hit predicted\n')


def custom_apply_classifier(argu):
    """ Adapt classifier application to input args. """
    # To be conservative: treat our test data as foreground and consider the best potential motifs
    motif_hits = get_motif_hits(argu, argu.in_fasta, True)
    if motif_hits:
        apply_classifier(argu, motif_hits, argu.feature_vector_type, argu.seq_feature)
    else:
        with open(argu.output, 'w') as stream:
            stream.write('No motif hits predicted\n')


# ********************
# BATCH EXECUTIONS
# ********************


def custom_train_and_apply_classifier(argu):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    # ********************
    # TRAIN CLASSIFIER
    # ********************

    fg_hits = get_motif_hits(argu, argu.fg_fasta, True)
    bg_hits = get_motif_hits(argu, argu.bg_fasta, False)

    fg_shapes = get_motif_dna_shapes_vector(fg_hits, argu.fg_bed, argu.shape_first_order,
                                            argu.shape_second_order, argu.extension, argu.scaled)
    bg_shapes = get_motif_dna_shapes_vector(bg_hits, argu.bg_bed, argu.shape_first_order,
                                            argu.shape_second_order, argu.extension, argu.scaled)

    foreground_data = get_feature_vector(argu, argu.feature_vector_type, argu.seq_feature, fg_hits, fg_shapes)
    background_data = get_feature_vector(argu, argu.feature_vector_type, argu.seq_feature, bg_hits, bg_shapes)
    foreground_data, background_data = match_feature_vector_length(foreground_data, background_data)

    data, labels = construct_classifier_input(foreground_data, background_data)

    # Get array of active feature names
    feature_names = construct_feature_names_array(argu, len(fg_hits[0]), shape_feature_names)

    output_data_vectors(argu, feature_names, data, labels)

    # Machine learning estimator
    classifier = GradientBoostingClassifier()

    # Cross-validation parameter
    cv = StratifiedKFold(n_splits=5)

    # K_fold execution
    k_fold_classification(argu, data, labels, classifier, cv, feature_names)

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


def k_fold_classification(argu, data, labels, classifier, cv, feature_names):
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    """ Run Kfold classification on a machine learning classifier """

    # CONVERT TO NUMPY ARRAYS PRIOR TO EXECUTION
    data = np.array(data)
    labels = np.array(labels)

    ################
    # GRAPH PREP
    ################
    # line colors
    colors = cycle(['indigo', 'blue', 'darkorange', 'yellow', 'green'])
    # line width
    lw = 1
    # prepare prc/roc figure
    init_basic_plot_settings(colors, lw)

    ###################
    # PRC START VALUES
    ###################
    reversed_mean_precision = 0.0
    mean_recall = np.linspace(0, 1, 100)
    init_prc_params(reversed_mean_precision, mean_recall)

    ###################
    # ROC START VALUES
    ###################
    # y-axis (dependent variable) = sensitivity
    mean_tpr = 0.0
    # x-axis (independent variable) = specificity
    mean_fpr = np.linspace(0, 1, 100)
    init_roc_params(mean_tpr, mean_fpr)

    # Initialize the two plots
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)
    fig.set_size_inches(10.5, 16.5)
    prc = fig.add_subplot(211)
    roc = fig.add_subplot(212)
    init_prc_and_roc_figure(fig, prc, roc)

    # Perform classification and plot PRC + ROC curve for each fold
    i = 0

    for (train, test), color in zip(cv.split(data, labels), colors):
        probas_ = classifier.fit(data[train], labels[train]).predict_proba(data[test])

        ################
        # PRC SECTION
        ################
        precision, recall, thresholds = precision_recall_curve(labels[test], probas_[:, 1])
        add_single_fold_prc_to_figure(precision, recall, color, i)

        ################
        # ROC SECTION
        ################
        fpr, tpr, thresholds = roc_curve(labels[test], probas_[:, 1])
        add_single_fold_roc_to_figure(fpr, tpr, color, i)

        i += 1

    n_splits = cv.get_n_splits(data, labels)
    mean_auprc = aggregate_k_prc_folds(argu, n_splits)
    mean_auroc = aggregate_k_roc_folds(argu, n_splits)

    # OUTPUT PRC_ROC PNG & AUPRC/AUROC VALS FOR CURRENT PROTEIN
    output_k_fold_prc_roc_results(argu, mean_auprc, mean_auroc)

    # OUTPUT FEATURE IMPORTANCES RANKING FOR CURRENT PROTEIN
    output_classifier_feature_importances(argu, classifier, data, labels, feature_names)


##############################################################################
#                               MAIN
##############################################################################


if __name__ == "__main__":
    arguments = arg_parsing()
    arguments.func(arguments)
