

def construct_seq_and_dna_shape_feature_vector(motif_hits, dna_shapes_matrix, extension=0, is_binary_encoding=False):
    """ Combine DNA sequence and shape features.
    The hit scores (PSSM or TFFM) or 4-bits encoding are combined with DNAshape
    in output feature vector for classification.
    """
    comb = []
    motif_hit_index = -1
    for hit in motif_hits:
        if hit:
            motif_hit_index += 1
            if dna_shapes_matrix:
                hit_shapes = []
                for shape_type_index in xrange(len(dna_shapes_matrix)):
                    hit_shapes += dna_shapes_matrix[shape_type_index][motif_hit_index]
                if (not is_binary_encoding and
                        (len(hit_shapes) ==
                            len(dna_shapes_matrix) * (len(hit) + 2 * extension))):
                    comb.append([hit.score] + hit_shapes)
                elif (is_binary_encoding and
                        (len(hit_shapes) ==
                            len(dna_shapes_matrix) * (len(hit) / 4 + 2 * extension))):
                    comb.append(hit + hit_shapes)
            elif is_binary_encoding:
                comb.append(hit)
            else:
                comb.append([hit.score])
    return comb


def construct_dna_shape_only_feature_vector(motif_hits, dna_shapes_matrix, extension=0, is_binary_encoding=False):
    """ Construct a feature vector for DNA shape only.
    The DNA shape values for each of our motif hits are consolidated
    in an output feature vector for classification.
    """
    comb = []
    motif_hit_index = -1
    for hit in motif_hits:
        if hit:
            motif_hit_index += 1
            if dna_shapes_matrix:
                hit_shapes = []
                for shape_type_index in xrange(len(dna_shapes_matrix)):
                    hit_shapes += dna_shapes_matrix[shape_type_index][motif_hit_index]
                if (not is_binary_encoding and
                        (len(hit_shapes) ==
                            len(dna_shapes_matrix) * (len(hit) + 2 * extension))):
                    comb.append(hit_shapes)
                elif (is_binary_encoding and
                        (len(hit_shapes) ==
                            len(dna_shapes_matrix) * (len(hit) / 4 + 2 * extension))):
                    comb.append(hit + hit_shapes)
            elif is_binary_encoding:
                comb.append(hit)
            else:
                comb.append([hit.score])
    return comb


def construct_dna_shape_and_flexibility_feature_vector(motif_hits, dna_shapes_matrix, promoter_region_flex_vector):
    """ Construct a feature vector combining DNA shape and promoter region flexibility.
    The DNA shape values for each of our motif hits are combined with the promoter region
    flexibility array to form a single output feature vector for classification.
    """
    comb = []
    motif_hit_index = -1
    for hit in motif_hits:
        if hit:
            motif_hit_index += 1
            if dna_shapes_matrix:
                hit_shapes = []
                for shape_type_index in xrange(len(dna_shapes_matrix)):
                    hit_shapes += dna_shapes_matrix[shape_type_index][motif_hit_index]
                if ((len(hit_shapes) ==
                            len(dna_shapes_matrix) * (len(hit)))):
                    # NOTE: flex_evals[motif_hit_index] is always a 1d array
                    comb.append(hit_shapes + promoter_region_flex_vector[motif_hit_index])
    return comb
