# GLOBAL IMPORTS
import os
import sys

# LIBRARY IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from Bio import trie
from Bio import triefind
from sklearn.metrics import auc

from DNAshapedTFBS_constants import *

from itertools import cycle

PATH = os.path.dirname(os.path.realpath(__file__))


# LOGISTICAL HELPER FUNCTIONS


def scale01(values, mini=None, maxi=None, tol=1e-6):
    """ Scale the values in [0, 1]. """
    from numpy import amax, amin
    if not mini:
        mini = amin(values)
    if not maxi:
        maxi = amax(values)
    scaled_values = [(val - mini) / (maxi - mini + tol) for val in values]
    return scaled_values, mini, maxi


def not_na(item):
    """ Remove NAs and empty values. """
    return not (item == "NA" or item == "")


def contains_zero(motif):
    """ Return True if the PSSM contains a 0 frequency at one position. """
    for nucleotide in 'ACGT':
        for count in motif.counts[nucleotide]:
            if count == 0.:
                return True
    return False


def encode_hits(hits):
    """
    Encode the sequence at hits using a binary encoding (4bits per nucleotide).

    hits corresponds to a list of HIT (TFFM module) instances.

    """
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
               'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    encoding = []
    for hit in hits:
        encoding.append(
            [val for nucl in hit.sequence() for val in mapping[nucl]])
    return encoding


def match_feature_vector_length(foreground_data, background_data):
    """
    Trim the foreground and background feature vectors to the same size

    """
    fg_len = len(foreground_data)
    bg_len = len(background_data)
    if fg_len > bg_len:
        foreground_data = foreground_data[0:bg_len]
    elif bg_len > fg_len:
        background_data = background_data[0:fg_len]
    return foreground_data, background_data


def get_positions_from_bed(bed_file):
    """ Get the positions of the sequences described in the bed file. """
    with open(bed_file) as stream:
        positions = {}
        for line in stream:
            spl = line.split()
            positions[spl[3]] = (spl[0], eval(spl[1]) + 1, eval(spl[2]))
    return positions


def get_jaspar_pssm(jaspar, bool_id=True):
    """ 

    Construct the PSSM from the JASPAR ID or JASPAR formatted file.

    We assume that we are using profiles from the CORE JASPAR db when providing
    a JASPAR ID. Hence the JASPAR ID should starts with 'MA'.
    If a filename is provided, we assume that the TF binding profile is using
    the JASPAR format as documented in the Bio.motifs.jaspar BioPython module.

    """
    import Bio.motifs
    if bool_id:
        from Bio.motifs.jaspar.db import JASPAR5
        # Please put your local JASPAR database information below
        jaspar_db_host = ""
        jaspar_db_name = ""
        jaspar_db_user = ""
        jaspar_db_pass = ""
        jdb = JASPAR5(host=jaspar_db_host, name=jaspar_db_name,
                      user=jaspar_db_user, password=jaspar_db_pass)
        motif = jdb.fetch_motif_by_id(jaspar)
        motif.pseudocounts = Bio.motifs.jaspar.calculate_pseudocounts(motif)
    else:
        with open(jaspar) as handle:
            motif = Bio.motifs.read(handle, 'jaspar')
            # If the PFM contains a zero, need to use pseudocounts
            if contains_zero(motif):
                import sys
                # The pseudocount will be minimal
                motif.pseudocounts = sys.float_info.min
    return motif.pssm


# FEATURE VECTORS I/O


def feature_vector_type_to_string(feature_vector_type):
    if feature_vector_type == SEQ_AND_DNA_SHAPE_TYPE_CONSTANT:
        return 'seq_and_dna_shape'
    elif feature_vector_type == DNA_SHAPE_ONLY_TYPE_CONSTANT:
        return 'dna_shape_only'
    elif feature_vector_type == DNA_SHAPE_AND_FLEX_TYPE_CONSTANT:
        return 'dna_shape_and_flex'
    elif feature_vector_type == SEQ_AND_FLEX_TYPE_CONSTANT:
        return 'seq_and_flex'
    elif feature_vector_type == FLEX_ONLY_TYPE_CONSTANT:
        return 'flex_only'


def seq_feature_type_to_string(seq_feature_type):
    if seq_feature_type == PSSM_SCORE_TYPE_CONSTANT:
        return 'pssm_only'
    elif seq_feature_type == TFFM_SCORE_TYPE_CONSTANT:
        return 'tffm_only'
    elif seq_feature_type == BINARY_ENCODING_TYPE_CONSTANT:
        return 'seq_binary_encoding_only'


def all_feature_names():
    feature_names = []

    # Append sequence feature
    feature_names += ['Seq_Feature_Value']

    for shapeName in SHAPE_FEATURE_NAMES:
        for position in xrange(MAX_MOTIF_LENGTH):
            feature_names += [shapeName + ' - ' + str(position)]

    flexibility_eval_function_str = ['Flex_Eval_Function']
    feature_names += flexibility_eval_function_str
    tri_nuc_classes = ['AAT', 'AAA', 'CCA', 'AAC', 'ACT', 'CCG', 'ATC', 'AAG', 'CGC', 'AGG', 'GAA', 'ACG', 'ACC',
                       'GAC', 'CCC', 'ACA', 'CGA', 'GGA', 'CAA', 'AGC', 'GTA', 'AGA', 'CTC', 'CAC', 'TAA', 'GCA',
                       'CTA', 'GCC', 'ATG', 'CAG', 'ATA', 'TCA']
    feature_names += tri_nuc_classes

    return feature_names


def format_data_instance(argu, motif_length, data_instance):
    is_eval_f = \
        True if argu.feature_vector_type == DNA_SHAPE_AND_FLEX_TYPE_CONSTANT \
                and argu.is_eval_f else False

    formatted_data_instance = [''] * ALL_FEATURES_COUNT
    formatted_data_instance_index = 0
    data_instance_index = 0
    feature_vector_type = argu.feature_vector_type
    if feature_vector_type not in SEQ_FEATURE_INCLUDED_CONSTANTS:
        formatted_data_instance[0] = 'N/A'
        formatted_data_instance_index += 1
    else:
        formatted_data_instance[0] = data_instance[0]
        formatted_data_instance_index += 1
        data_instance_index += 1

    if feature_vector_type in DNA_SHAPE_FEATURE_TYPE_CONSTANTS:
        for shapeName in SHAPE_FEATURE_NAMES:
            for position in xrange(motif_length):
                formatted_data_instance[formatted_data_instance_index] = data_instance[data_instance_index]
                formatted_data_instance_index += 1
                data_instance_index += 1
            for position in xrange(MAX_MOTIF_LENGTH - motif_length):
                formatted_data_instance[formatted_data_instance_index] = 'N/A'
                formatted_data_instance_index += 1
    else:
        for shapeName in SHAPE_FEATURE_NAMES:
            for position in xrange(MAX_MOTIF_LENGTH):
                formatted_data_instance[formatted_data_instance_index] = 'N/A'
                formatted_data_instance_index += 1

    if feature_vector_type in FLEXIBILITY_TYPE_CONSTANTS:
        if is_eval_f:  # we used the eval function feature
            formatted_data_instance[formatted_data_instance_index] = data_instance[data_instance_index]
            formatted_data_instance_index += 1
            data_instance_index += 1
            for position in xrange(len(TRI_NUC_CLASSES)):
                formatted_data_instance[formatted_data_instance_index] = 'N/A'
                formatted_data_instance_index += 1
        else:  # we used the trinucleotide counts directly
            formatted_data_instance[formatted_data_instance_index] = 'N/A'
            formatted_data_instance_index += 1
            for position in xrange(len(TRI_NUC_CLASSES)):
                formatted_data_instance[formatted_data_instance_index] = data_instance[data_instance_index]
                formatted_data_instance_index += 1
                data_instance_index += 1
    else:
        formatted_data_instance[formatted_data_instance_index] = 'N/A'
        formatted_data_instance_index += 1
        for position in xrange(len(TRI_NUC_CLASSES)):
            formatted_data_instance[formatted_data_instance_index] = 'N/A'
            formatted_data_instance_index += 1

    return formatted_data_instance


def output_experimental_results(argu, predictions, motif_length, feature_vector_type,
                                seq_feature_type, feature_names, data, labels):
    import csv

    peak_ids = None
    peak_start = None
    peak_end = None
    peak_strand = None
    peak_sequence = None
    proba = None
    if predictions:
        peak_ids = predictions['peak_id']
        peak_start = predictions['start']
        peak_end = predictions['end']
        peak_strand = predictions['strand']
        peak_sequence = predictions['sequence']
        proba = predictions['proba']

    # Write data to protein-specific file
    # protein, feature_vector, label
    csv_title = argu.output + '_DATA_INSTANCES.csv'
    with open(r'' + csv_title, 'w') as f:
        writer = csv.writer(f)
        title_headers = ['Protein', 'Prior_Classification', 'Predicted_Binding_Probability'] + feature_names
        writer.writerow(title_headers)
        i = 0
        for data_instance in data:
            writer = csv.writer(f)
            label = labels[i]
            label_str = 'NotBound' if label == 0 else 'Bound'
            try:
                protein_name = argu.protein
            except AttributeError:
                protein_name = argu.output
            predicted_binding_probability = 'N/A' if not predictions else proba[i]
            fields = [protein_name, label_str, predicted_binding_probability] + data_instance
            i += 1
            writer.writerow(fields)
    # Append data to cumulative experiments file
    if not os.path.isfile(CUMULATIVE_EXPERIMENTS_PATH):
        with open(r'' + CUMULATIVE_EXPERIMENTS_PATH, 'w') as f:
            writer = csv.writer(f)
            prediction_headers = ['Peak_Id', 'Peak_Start_Offset', 'Peak_End_Offset',
                                  'Strand', 'Sequence', 'Predicted_Binding_Probability']
            title_headers = ['Experiment_Name', 'Feature_Vector_Type', 'Sequence_Feature_Type', 'Background_Type',
                             'Protein', 'Prior_Classification'] + prediction_headers + all_feature_names()
            writer.writerow(title_headers)
    with open(r'' + CUMULATIVE_EXPERIMENTS_PATH, 'a') as f:
        i = 0
        for data_instance in data:
            writer = csv.writer(f)
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
            predictions_data = ['N/A'] * 6
            if predictions:
                predictions_data = [peak_ids[i], peak_start[i], peak_end[i],
                                    peak_strand[i], peak_sequence[i], proba[i]]
            label = labels[i]
            label_str = 'NotBound' if label == 0 else 'Bound'
            try:
                protein_name = argu.protein
            except AttributeError:
                protein_name = argu.output
            fields = titles + [protein_name, label_str] + predictions_data +\
                format_data_instance(argu, motif_length, data_instance)
            i += 1
            writer.writerow(fields)


# PREDICTIONS I/O


def make_predictions(clf, tests, hits, proba_threshold):
    """ Predict hits from the tests using the classifier. """
    predictions = {'peak_id': [], 'start': [], 'end': [], 'strand': [],
                   'sequence': [], 'proba': []}
    for indx, proba in enumerate(clf.predict_proba(tests)):
        if proba[1] >= proba_threshold:
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


def output_classifier_predictions(predictions, output):
    """ Print the predictions in the output file. """
    import pandas as pd
    pd_predictions = pd.DataFrame(predictions)
    pd.set_option('display.max_rows', len(pd_predictions))
    with open(output, 'w') as stream:
        stream.write('{0}\n'.format(pd_predictions.to_string(
            index=False, columns=['peak_id', 'start', 'end', 'strand',
                                  'sequence', 'proba'])))


# ROC/PRC FILE I/O

# global variable defaults
colors = cycle(['indigo', 'blue', 'darkorange', 'yellow', 'green'])
lw = 1
# prc params
reversed_mean_precision = 0.0
mean_recall = np.linspace(0, 1, 100)
# roc params
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
# default initialized figure
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

    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    global reversed_mean_precision

    reversed_recall = np.fliplr([recall])[0]
    reversed_precision = np.fliplr([precision])[0]
    reversed_mean_precision += interp(mean_recall, reversed_recall, reversed_precision)
    reversed_mean_precision[0] = 0.0

    prc_auc = auc(recall, precision)
    prc.plot(recall, precision, lw=lw, color=color,
             label='PRC fold %d (area = %0.6f)' % (fold_number, prc_auc))


def add_single_fold_roc_to_figure(fpr, tpr, color, fold_number):
    import numpy as np
    from scipy import interp

    from sklearn.metrics import roc_curve, auc, precision_recall_curve

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


# FEATURE IMPORTANCE I/O


def construct_feature_names_array(argu, motif_length, shape_feature_names):
    print "\n\nOur shape features:", shape_feature_names
    print "\n\nOur motif length:", motif_length
    is_eval_f = \
        True if argu.feature_vector_type == DNA_SHAPE_AND_FLEX_TYPE_CONSTANT \
                and argu.is_eval_f else False

    feature_names = []
    feature_vector_type = argu.feature_vector_type
    if feature_vector_type in SEQ_FEATURE_INCLUDED_CONSTANTS:
        seq_feature = argu.seq_feature
        if seq_feature == PSSM_SCORE_TYPE_CONSTANT:  # PSSM
            feature_names += ['PSSM_SCORE']
        elif seq_feature == TFFM_SCORE_TYPE_CONSTANT:  # TFFM
            feature_names += ['TFFM_SCORE']
        elif seq_feature == BINARY_ENCODING_TYPE_CONSTANT:  # Binary encoding
            feature_names += ['SEQUENCE_ENCODING']

    if feature_vector_type in DNA_SHAPE_FEATURE_TYPE_CONSTANTS:
        for shapeName in shape_feature_names:
            for position in xrange(motif_length):
                feature_names += [shapeName + ' - ' + str(position)]

    if feature_vector_type in FLEXIBILITY_TYPE_CONSTANTS:
        if is_eval_f:  # we used the eval function feature
            flexibility_eval_function_str = ['Flex_Eval_Function']
            feature_names += flexibility_eval_function_str
        else:  # we used the trinucleotide counts directly
            tri_nuc_classes = ['AAT', 'AAA', 'CCA', 'AAC', 'ACT', 'CCG', 'ATC', 'AAG', 'CGC', 'AGG', 'GAA', 'ACG',
                               'ACC',
                               'GAC', 'CCC', 'ACA', 'CGA', 'GGA', 'CAA', 'AGC', 'GTA', 'AGA', 'CTC', 'CAC', 'TAA',
                               'GCA',
                               'CTA', 'GCC', 'ATG', 'CAG', 'ATA', 'TCA']
            feature_names += tri_nuc_classes

    return feature_names


def output_classifier_feature_importances(argu, classifier, data, feature_names):
    import datetime as dt
    import numpy as np
    import csv
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    with open(r'' + argu.output + '_FEATURE_IMPORTANCES.csv', 'w') as feature_importances_file:
        writer = csv.writer(feature_importances_file)
        headers = ['Day - Hour', 'Protein', 'Feature_Name', 'Importance_Value']
        writer.writerow(headers)
    # NOTE: data.shape[1] below is a call to numpy for the dimension m of our n x m data matrix
    for row_number in range(data.shape[1]):
        date_hour = '{}'.format(dt.datetime.today().day) + ' - ' + '{}'.format(dt.datetime.today().hour)
        # date-hour, protein, featureName, importance
        try:
            protein_name = argu.protein
        except AttributeError:
            protein_name = argu.output
        fields = [date_hour, protein_name, feature_names[indices[row_number] - 1], importances[indices[row_number]]]
        with open(r'' + argu.output + '_FEATURE_IMPORTANCES.csv', 'a') as feature_importances_file:
            writer = csv.writer(feature_importances_file)
            writer.writerow(fields)


# DETERMINISTIC MOTIF SCANNING


def get_motif_hits(argu, seq_file, is_foreground):
    seq_feature_type = argu.seq_feature
    if seq_feature_type == PSSM_SCORE_TYPE_CONSTANT \
            or seq_feature_type == BINARY_ENCODING_TYPE_CONSTANT:  # PSSM or Encoding
        if argu.jasparid:
            pssm = get_jaspar_pssm(argu.jasparid)
        else:
            pssm = get_jaspar_pssm(argu.jasparfile, False)
        return find_pssm_hits(pssm, seq_file, is_foreground)
    elif seq_feature_type == TFFM_SCORE_TYPE_CONSTANT:  # TFFM
        return find_tffm_hits(argu.tffm_file, seq_file, argu.tffm_kind)


def find_pssm_hits(pssm, seq_file, is_foreground):
    """ Predict hits in sequences using a PSSM. """
    from operator import itemgetter
    import math
    import Bio.SeqIO
    from Bio.Alphabet import generic_dna
    from Bio.Alphabet.IUPAC import IUPACUnambiguousDNA as unambiguousDNA
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


def find_tffm_hits(xml, seq_file, tffm_kind):
    """ Predict hits in sequences using a TFFM. """
    # TODO: Test if TFFM is installed instead of using local env.
    sys.path.append('{0}/TFFM/'.format(PATH))

    import tffm_module
    from DNAshapedTFBS_constants import TFFM_KIND  # TFFM-framework
    if tffm_kind == 'first_order':
        tffm_kind = TFFM_KIND.FIRST_ORDER
    elif tffm_kind == 'detailed':
        tffm_kind = TFFM_KIND.DETAILED
    else:
        sys.exit('The type of TFFM should be "first_order" or "detailed".')
    tffm = tffm_module.tffm_from_xml(xml, tffm_kind)
    return [hit for hit in
            tffm.scan_sequences(seq_file, only_best=True) if hit]


# EXTENDED MOTIF SCANNING


def extended_hit_pos(hit, peak_chrom, peak_start, extension=0):
    """ Extend the hit by 'extension' nt to compute DNAshape features. """
    start = peak_start + hit.start - extension - 2  # BED
    end = peak_start + hit.end + extension - 1  # BED
    return peak_chrom, start, end


def output_extended_motif_hits(hits, positions, extension=0):
    """
    Write the extended hits to a temporary bed file.

    :returns: The name of the temporary file.
    :rtype: str

    """

    import tempfile
    import os
    fdescr, tmp_file = tempfile.mkstemp()
    os.close(fdescr)
    with open(tmp_file, 'w') as stream:
        for hit in hits:
            if hit:
                identifier = hit.seq_record.id
                peak_chrom, peak_start, _ = positions[identifier]
                chrom, start, end = extended_hit_pos(hit, peak_chrom,
                                                     peak_start, extension)
                if not chrom.startswith("chr"):
                    chrom = "chr{0}".format(chrom)
                if 0.0 <= hit.score <= 1.0:
                    stream.write("{0}\t{1:d}\t{2:d}\t{3}\t{4:d}\t{5}\n".format(
                        chrom, start, end, identifier, int(hit.score * 100),
                        hit.strand))
                else:
                    stream.write("{0}\t{1:d}\t{2:d}\t{3}\t{4:d}\t{5}\n".format(
                        chrom, start, end, identifier, 0,
                        hit.strand))
    return tmp_file


# HELPER FUNCTIONS FOR DNA SHAPE EVALUATION


def get_score_of_dna_shape(in_file, shape=None, scaled=False):
    """ Get DNAshape values for particular geometry. """
    with open(in_file) as stream:
        scores = []
        for line in stream:
            values = [item for item in line.rstrip().split()[7].split(',')
                      if not_na(item)]
            values = [eval(value) for value in values]
            if scaled:
                mini, maxi = DNASHAPEINTER[shape]
                values, _, _ = scale01(values, mini, maxi)
            scores.append(values)
        return scores


def get_motif_dna_shapes_matrix(motif_hits, bed_file, shape_first_order,
                                shape_second_order, extension=0, scaled=False):
    """ Retrieve DNAshape feature values for the hits. """
    import subprocess
    import os
    # What shape feature are we currently considering?
    bigwigs = shape_first_order + shape_second_order
    print(bigwigs)
    # Retrieve peak from bed file
    peaks_pos = get_positions_from_bed(bed_file)
    with open(os.devnull, 'w') as devnull:
        tmp_file = output_extended_motif_hits(motif_hits, peaks_pos, extension)
        # TODO: put MGW2 back here
        # MODIFIED HERE TO REMOVE MGW2
        shapes = ['HelT', 'ProT', 'MGW', 'Roll', 'HelT2', 'ProT2',
                  'Roll2']
        dna_shapes_matrix = []
        for indx, bigwig in enumerate(bigwigs):
            if bigwig:
                out_file = '{0}.{1}'.format(tmp_file, shapes[indx])
                try:
                    subprocess.call([BWTOOL, 'ex', 'bed', tmp_file, bigwig, out_file],
                                    stdout=devnull)
                    print(out_file)
                except:
                    print("THERE WAS AN ERROR READING THIS BW FILE")
                if indx < 4:
                    dna_shapes_matrix.append(get_score_of_dna_shape(out_file, shapes[indx], scaled))
                else:
                    dna_shapes_matrix.append(get_score_of_dna_shape(out_file, shapes[indx]))
        subprocess.call(['rm', '-f', '{0}.HelT'.format(tmp_file),
                         '{0}.MGW'.format(tmp_file), '{0}.ProT'.format(tmp_file),
                         '{0}.Roll'.format(tmp_file), '{0}.HelT2'.format(tmp_file),
                         '{0}.ProT2'.format(tmp_file), '{0}.Roll2'.format(tmp_file), tmp_file])

        return dna_shapes_matrix


# HELPER FUNCTIONS FOR PROMOTER REGION FLEXIBILITY EVALUATION


def seq_splice(seq, w_start, w_end, ext_size):
    """  returns start and end index in spliced sequence """
    seq_length = len(seq)
    pos_start, pos_end = 0, 0
    if w_start >= ext_size:
        if w_end < (seq_length - ext_size):  # (w_start >= L) and (w_end <= len(S) - L)
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
    # RESTRICT INDICES IN CASE OF ERRORS ABOVE
    # FIXME: The below code is currently executing in some edge cases...
    if pos_start < 0:
        pos_start = 0
    if pos_end > (seq_length - 1):
        pos_end = seq_length - 1
    return pos_start, pos_end


def get_promoter_region_flex_matrix(motif_hits, is_eval_f):
    """ evaluate promoter region of hits """
    from itertools import product
    from Bio.Seq import Seq
    from Bio.Alphabet import generic_dna
    from Bio.Alphabet.IUPAC import IUPACUnambiguousDNA as unambiguousDNA
    # Build trie structure to evaluate promoter regions
    flexibility_key_words = ['AAT', 'AAA', 'CCA', 'AAC', 'ACT', 'CCG', 'ATC', 'AAG',
                             'CGC', 'AGG', 'GAA', 'ACG', 'ACC', 'GAC', 'CCC', 'ACA',
                             'CGA', 'GGA', 'CAA', 'AGC', 'GTA', 'AGA', 'CTC', 'CAC',
                             'TAA', 'GCA', 'CTA', 'GCC', 'ATG', 'CAG', 'ATA', 'TCA']
    trinucleotide_words = []
    if is_eval_f:  # eval function trie
        tr_eval = trie.trie()
        bending_propensities = [0.755783741, 0.760332075, 0.781922225, 0.814647316, 0.832768156,
                                0.872842632, 0.895834135, 0.922193691, 0.925889854, 0.944594069,
                                0.963676135, 0.96753856, 0.968506582, 0.987084135, 0.988071713,
                                0.994017964, 0.997004496, 1.013084867, 1.010050167, 1.017145322,
                                1.025315121, 1.027367803, 1.031485504, 1.040810774, 1.070365308,
                                1.078962574, 1.094174284, 1.112934254, 1.14339282, 1.191246217,
                                1.199614194, 1.214096283]
        for i in xrange(0, len(flexibility_key_words)):
            bending_propensity = bending_propensities[i]
            word = flexibility_key_words[i]
            word_seq_record = Seq(word, generic_dna)
            compl_word = str(word_seq_record.reverse_complement())
            tr_eval[word] = bending_propensity
            tr_eval[compl_word] = bending_propensity
    else:  # trinucleotide_counts trie
        # Enumerate all trinucleotide keys
        alphabet = unambiguousDNA()
        trinucleotide_words = [''.join(i) for i in product(alphabet.letters, repeat=3)]
    # Iteratively evaluate promoter regions of hits
    flex_matrix = []
    for hit in motif_hits:
        if hit:
            hit_seq = hit.seq_record.seq
            # print "Sequence length:", len(hit_seq)
            # print "(hit_start, hit_end) =", (hit.start, hit.end)
            ext_start, ext_end = seq_splice(hit_seq, hit.start, hit.end, 50)
            # print "Hit_seq:", hit_seq, "(start, end) =", (ext_start, ext_end)
            ext_seq = str(hit_seq[ext_start:ext_end + 1].upper())
            # print "Ext_seq:", ext_seq
            if is_eval_f:  # using eval function
                eval_result = 0.0
                for word in triefind.find(ext_seq, tr_eval):
                    eval_result += tr_eval[word[0]]
                print eval_result
                flex_matrix.append([eval_result])
            else:  # using trinucleotide_counts trie
                trinucleotide_counts = []
                tr_count = trie.trie()
                for key in trinucleotide_words:
                    tr_count[key] = 0
                for word in triefind.find(ext_seq, tr_count):
                    tr_count[word[0]] += 1
                for i in xrange(0, len(flexibility_key_words)):
                    word = flexibility_key_words[i]
                    word_seq_record = Seq(word, generic_dna)
                    compl_word = str(word_seq_record.reverse_complement())
                    # print "Sequence:", ext_seq
                    # print "Word:", word, ", count:", tr_count[word]
                    # print "Complement:", compl_word, ", count: ", tr_count[compl_word]
                    count = tr_count[word] + tr_count[compl_word]
                    # print "Count:", count
                    trinucleotide_counts.append(count)
                # print "Counts:", trinucleotide_counts
                flex_matrix.append(trinucleotide_counts)
    return flex_matrix
