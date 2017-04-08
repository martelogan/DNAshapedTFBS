import os
import csv

from DNAshapedTFBS_common_utils import feature_vector_type_to_string, seq_feature_type_to_string, all_feature_names
from DNAshapedTFBS_constants import *


# FEATURE VECTORS I/O


def format_data_instance(argu, motif_length, data_instance):
    is_eval_f = \
        True if argu.feature_vector_type == DNA_SHAPE_AND_FLEX_TYPE_CONSTANT \
        and argu.is_eval_f else False

    formatted_data_instance = [MISSING_DATA_STRING] * ALL_FEATURES_COUNT
    formatted_data_instance_index = 0
    data_instance_index = 0
    feature_vector_type = argu.feature_vector_type
    if feature_vector_type not in SEQ_FEATURE_INCLUDED_CONSTANTS:
        formatted_data_instance[0] = MISSING_DATA_STRING
        formatted_data_instance_index += 1
    else:
        formatted_data_instance[0] = data_instance[0]
        formatted_data_instance_index += 1
        data_instance_index += 1

    if feature_vector_type in DNA_SHAPE_FEATURE_TYPE_CONSTANTS:
        for shape_name in SHAPE_FEATURE_NAMES:
            for position in xrange(motif_length):
                formatted_data_instance[formatted_data_instance_index] = data_instance[data_instance_index]
                formatted_data_instance_index += 1
                data_instance_index += 1
            for position in xrange(MAX_MOTIF_LENGTH - motif_length):
                formatted_data_instance[formatted_data_instance_index] = MISSING_DATA_STRING
                formatted_data_instance_index += 1
    else:
        for shape_name in SHAPE_FEATURE_NAMES:
            for position in xrange(MAX_MOTIF_LENGTH):
                formatted_data_instance[formatted_data_instance_index] = MISSING_DATA_STRING
                formatted_data_instance_index += 1

    if feature_vector_type in FLEXIBILITY_TYPE_CONSTANTS:
        if is_eval_f:  # we used the eval function feature
            formatted_data_instance[formatted_data_instance_index] = data_instance[data_instance_index]
            formatted_data_instance_index += 1
            data_instance_index += 1
            for position in xrange(len(TRI_NUC_CLASSES)):
                formatted_data_instance[formatted_data_instance_index] = MISSING_DATA_STRING
                formatted_data_instance_index += 1
        else:  # we used the trinucleotide counts directly
            formatted_data_instance[formatted_data_instance_index] = MISSING_DATA_STRING
            formatted_data_instance_index += 1
            for position in xrange(len(TRI_NUC_CLASSES)):
                formatted_data_instance[formatted_data_instance_index] = data_instance[data_instance_index]
                formatted_data_instance_index += 1
                data_instance_index += 1
    else:
        formatted_data_instance[formatted_data_instance_index] = MISSING_DATA_STRING
        formatted_data_instance_index += 1
        for position in xrange(len(TRI_NUC_CLASSES)):
            formatted_data_instance[formatted_data_instance_index] = MISSING_DATA_STRING
            formatted_data_instance_index += 1

    return formatted_data_instance


def output_experimental_results(argu, predictions, motif_length, feature_vector_type,
                                seq_feature_type, feature_names, data, labels):

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
                background_type_str = MISSING_DATA_STRING
            titles = [exp_name, feature_vector_type_str, seq_feature_type_str, background_type_str]
            predictions_data = [MISSING_DATA_STRING] * 6
            if predictions:
                predictions_data = [peak_ids[i], peak_start[i], peak_end[i],
                                    peak_strand[i], peak_sequence[i], proba[i]]
            label = labels[i]
            label_str = 'Not_Bound' if label == 0 else 'Bound'
            try:
                protein_name = argu.protein
            except AttributeError:
                protein_name = argu.output
            fields = titles + [protein_name, label_str] + predictions_data + \
                format_data_instance(argu, motif_length, data_instance)
            i += 1
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
            tri_nuc_classes = TRI_NUC_CLASSES
            feature_names += tri_nuc_classes

    return feature_names


def output_classifier_feature_importances(argu, classifier, data, feature_names):
    import datetime as dt
    import numpy as np
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


# PREDICTIONS I/O


def make_predictions(clf, tests, hits, proba_threshold):
    """ Predict hits from the tests using fitted classifier. """
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


