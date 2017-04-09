# GLOBAL IMPORTS
import os
import sys

# LIBRARY IMPORTS
from Bio import trie
from Bio import triefind

from DNAshapedTFBS_constants import *

PATH = os.path.dirname(os.path.realpath(__file__))


# STRING LOGISTICS FUNCTIONS


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
    feature_names += TRI_NUC_CLASSES

    return feature_names


# DATA LOGISTICS FUNCTIONS


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


# DETERMINISTIC MOTIF SCANNING


def get_motif_hits(argu, seq_file, is_foreground):
    seq_feature_type = argu.seq_feature_type
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
        # MODIFIED HERE TO REMOVE MGW2
        shapes = SHAPE_FEATURE_NAMES
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
        subprocess.call(['rm', '-f', '{0}.HelT'.format(tmp_file), '{0}.ProT'.format(tmp_file),
                         '{0}.MGW'.format(tmp_file), '{0}.Roll'.format(tmp_file),
                         '{0}.HelT2'.format(tmp_file), '{0}.ProT2'.format(tmp_file),
                         '{0}.MGW2'.format(tmp_file), '{0}.Roll2'.format(tmp_file), tmp_file])

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
    flexibility_scale_words = TRI_NUC_CLASSES
    all_trinucleotide_words = []
    if is_eval_f:  # eval function trie
        tr_eval = trie.trie()
        bending_propensities = TRINUC_BENDING_PROPENSITIES
        for i in xrange(0, len(flexibility_scale_words)):
            bending_propensity = bending_propensities[i]
            word = flexibility_scale_words[i]
            word_seq_record = Seq(word, generic_dna)
            compl_word = str(word_seq_record.reverse_complement())
            tr_eval[word] = bending_propensity
            tr_eval[compl_word] = bending_propensity
    else:  # trinucleotide_counts trie
        # Enumerate all trinucleotide keys
        alphabet = unambiguousDNA()
        all_trinucleotide_words = [''.join(i) for i in product(alphabet.letters, repeat=3)]
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
                for key in all_trinucleotide_words:
                    tr_count[key] = 0
                for word in triefind.find(ext_seq, tr_count):
                    tr_count[word[0]] += 1
                for i in xrange(0, len(flexibility_scale_words)):
                    word = flexibility_scale_words[i]
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