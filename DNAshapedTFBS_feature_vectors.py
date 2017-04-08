from DNAshapedTFBS_common_utils import *
from DNAshapedTFBS_constants import *


def construct_seq_and_dna_shape_feature_vectors(motif_hits, dna_shapes_matrix, extension=0, is_binary_encoding=False):
    """ Combine DNA sequence and shape features.
    The hit scores (PSSM or TFFM) or 4-bits encoding are combined with DNAshape
    in output array of feature vectors for classification.
    """
    feature_vectors = []
    motif_hit_index = -1
    for hit in motif_hits:
        if hit:
            motif_hit_index += 1
            if dna_shapes_matrix:
                feature_vector = []
                for shape_type_index in xrange(len(dna_shapes_matrix)):
                    feature_vector += dna_shapes_matrix[shape_type_index][motif_hit_index]
                if (not is_binary_encoding and
                        (len(feature_vector) ==
                            len(dna_shapes_matrix) * (len(hit) + 2 * extension))):
                    feature_vectors.append([hit.score] + feature_vector)
                elif (is_binary_encoding and
                        (len(feature_vector) ==
                            len(dna_shapes_matrix) * (len(hit) / 4 + 2 * extension))):
                    feature_vectors.append(hit + feature_vector)
            elif is_binary_encoding:
                feature_vectors.append(hit)
            else:
                feature_vectors.append([hit.score])
    return feature_vectors


def construct_seq_and_flexibility_feature_vectors(motif_hits, promoter_region_flex_matrix, is_binary_encoding=False):
    """ Combine DNA sequence and promoter region flexibility features.
    The hit scores (PSSM or TFFM) or 4-bits encoding are combined with flexibility
    in output array of feature vectors for classification.
    """
    feature_vectors = []
    motif_hit_index = -1
    for hit in motif_hits:
        if hit:
            motif_hit_index += 1
            if promoter_region_flex_matrix:
                if not is_binary_encoding:
                    # NOTE: promoter_region_flex_matrix[motif_hit_index] is always a 1d array
                    feature_vectors.append(hit.score + promoter_region_flex_matrix[motif_hit_index])
                elif is_binary_encoding:
                    # NOTE: promoter_region_flex_matrix[motif_hit_index] is always a 1d array
                    feature_vectors.append(hit + promoter_region_flex_matrix[motif_hit_index])
            elif is_binary_encoding:
                feature_vectors.append(hit)
            else:
                feature_vectors.append([hit.score])
    return feature_vectors


def construct_dna_shape_only_feature_vectors(motif_hits, dna_shapes_matrix):
    """ Construct feature vectors for DNA shape only.
    The DNA shape values for each of our motif hits are consolidated
    in output array of feature vectors for classification.
    """
    feature_vectors = []
    motif_hit_index = -1
    for hit in motif_hits:
        if hit:
            motif_hit_index += 1
            if dna_shapes_matrix:
                feature_vector = []
                for shape_type_index in xrange(len(dna_shapes_matrix)):
                    feature_vector += dna_shapes_matrix[shape_type_index][motif_hit_index]
                feature_vectors.append(feature_vector)
    return feature_vectors


def construct_flexibility_only_feature_vectors(motif_hits, promoter_region_flex_matrix):
    """ Construct feature vectors for promoter region flexibility only.
    The DNA shape values for each of our motif hits are consolidated
    in output array of feature vectors for classification.
    """
    feature_vectors = []
    motif_hit_index = -1
    for hit in motif_hits:
        if hit:
            motif_hit_index += 1
            if promoter_region_flex_matrix:
                feature_vectors.append(promoter_region_flex_matrix[motif_hit_index])
    return feature_vectors


def construct_dna_shape_and_flexibility_feature_vector(motif_hits, dna_shapes_matrix, promoter_region_flex_matrix):
    """ Construct feature vectors combining DNA shape and promoter region flexibility.
    The DNA shape values for each of our motif hits are combined with the promoter region
    flexibility array to form a single output array of feature vectors for classification.
    """
    feature_vectors = []
    motif_hit_index = -1
    for hit in motif_hits:
        if hit:
            motif_hit_index += 1
            if dna_shapes_matrix:
                feature_vector = []
                for shape_type_index in xrange(len(dna_shapes_matrix)):
                    feature_vector += dna_shapes_matrix[shape_type_index][motif_hit_index]
                if ((len(feature_vector) ==
                            len(dna_shapes_matrix) * (len(hit)))):
                    # NOTE: promoter_region_flex_matrix[motif_hit_index] is always a 1d array
                    feature_vectors.append(feature_vector + promoter_region_flex_matrix[motif_hit_index])
    return feature_vectors


def get_feature_vectors(argu, feature_vector_type, seq_feature_type, motif_hits, dna_shapes_matrix):
    bool4bits = True if seq_feature_type == BINARY_ENCODING_TYPE_CONSTANT else False
    if feature_vector_type == SEQ_AND_DNA_SHAPE_TYPE_CONSTANT:  # DNA Shape + Sequence
        return construct_seq_and_dna_shape_feature_vectors(motif_hits, dna_shapes_matrix, argu.extension, bool4bits)
    elif feature_vector_type == DNA_SHAPE_ONLY_TYPE_CONSTANT:  # DNA Shape Only
        return construct_dna_shape_only_feature_vectors(motif_hits, dna_shapes_matrix)
    elif feature_vector_type == DNA_SHAPE_AND_FLEX_TYPE_CONSTANT:  # DNA Shape + Flex
        is_eval_f = True if argu.is_eval_f else False
        flex_matrix = get_promoter_region_flex_matrix(motif_hits, is_eval_f)
        return construct_dna_shape_and_flexibility_feature_vector(motif_hits, dna_shapes_matrix, flex_matrix)
    elif feature_vector_type == SEQ_AND_FLEX_TYPE_CONSTANT:  # Flexibility + Sequence
        is_eval_f = True if argu.is_eval_f else False
        flex_matrix = get_promoter_region_flex_matrix(motif_hits, is_eval_f)
        return construct_seq_and_flexibility_feature_vectors(motif_hits, flex_matrix, bool4bits)
    elif feature_vector_type == FLEX_ONLY_TYPE_CONSTANT:  # Flexibility Only
        is_eval_f = True if argu.is_eval_f else False
        flex_matrix = get_promoter_region_flex_matrix(motif_hits, is_eval_f)
        return construct_flexibility_only_feature_vectors(motif_hits, flex_matrix)
