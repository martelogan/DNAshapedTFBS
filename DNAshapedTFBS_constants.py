# PATH VARIABLES
# TODO: get these from bash-parseable config file
BWTOOL = "bwtool"
CUMULATIVE_EXPERIMENTS_PATH = "CUMULATIVE_EXPERIMENTS.csv"
CUMULATIVE_AUPRC_AUROC_PATH = "CUMULATIVE_AUPRC_AUROC.csv"

# FEATURE CONSTANTS
MAX_MOTIF_LENGTH = 20
# TODO: use this internally and update with final realization of bigwigs
SHAPE_FEATURE_NAMES = ['HelT', 'ProT', 'MGW', 'Roll', 'HelT2', 'ProT2', 'MGW2', 'Roll2']
TRI_NUC_CLASSES = ['AAT', 'AAA', 'CCA', 'AAC', 'ACT', 'CCG', 'ATC', 'AAG', 'CGC', 'AGG', 'GAA', 'ACG', 'ACC',
                   'GAC', 'CCC', 'ACA', 'CGA', 'GGA', 'CAA', 'AGC', 'GTA', 'AGA', 'CTC', 'CAC', 'TAA', 'GCA',
                   'CTA', 'GCC', 'ATG', 'CAG', 'ATA', 'TCA']

# Status: 1 sequence feature + (MOTIF_LENGTH * NUM_SHAPES) shape features + Flex_Eval + (64 Trinuc_Words)
ALL_FEATURES_COUNT = 1 + (MAX_MOTIF_LENGTH * len(SHAPE_FEATURE_NAMES)) + 1 + len(TRI_NUC_CLASSES)

# FEATURE VECTOR TYPE CONSTANTS
DNA_SHAPE_ONLY_TYPE_CONSTANT = 0
FLEX_ONLY_TYPE_CONSTANT = 1
DNA_SHAPE_AND_FLEX_TYPE_CONSTANT = 2
SEQ_AND_DNA_SHAPE_TYPE_CONSTANT = 3
SEQ_AND_FLEX_TYPE_CONSTANT = 4

# SEQUENCE FEATURE TYPE CONSTANTS
PSSM_SCORE_TYPE_CONSTANT = 0
TFFM_SCORE_TYPE_CONSTANT = 1
BINARY_ENCODING_TYPE_CONSTANT = 2

# GROUPED CONSTANTS
DNA_SHAPE_FEATURE_TYPE_CONSTANTS = {SEQ_AND_DNA_SHAPE_TYPE_CONSTANT, DNA_SHAPE_ONLY_TYPE_CONSTANT,
                                    DNA_SHAPE_AND_FLEX_TYPE_CONSTANT}
FLEXIBILITY_TYPE_CONSTANTS = {SEQ_AND_FLEX_TYPE_CONSTANT, FLEX_ONLY_TYPE_CONSTANT,
                              DNA_SHAPE_AND_FLEX_TYPE_CONSTANT}
SEQ_FEATURE_INCLUDED_CONSTANTS = {SEQ_AND_FLEX_TYPE_CONSTANT, SEQ_AND_FLEX_TYPE_CONSTANT}

# PYTHON DATA CONSTANTS
DNASHAPEINTER = {'HelT': (30.94, 38.05), 'MGW': (2.85, 6.20),
                 'ProT': (-16.51, -0.03), 'Roll': (-8.57, 8.64)}
# TODO: change below to Young's modulus
TRINUC_BENDING_PROPENSITIES = [0.755783741, 0.760332075, 0.781922225, 0.814647316, 0.832768156,
                               0.872842632, 0.895834135, 0.922193691, 0.925889854, 0.944594069,
                               0.963676135, 0.96753856, 0.968506582, 0.987084135, 0.988071713,
                               0.994017964, 0.997004496, 1.013084867, 1.010050167, 1.017145322,
                               1.025315121, 1.027367803, 1.031485504, 1.040810774, 1.070365308,
                               1.078962574, 1.094174284, 1.112934254, 1.14339282, 1.191246217,
                               1.199614194, 1.214096283]

# LOGISTICAL STRING CONSTANTS
MISSING_DATA_STRING = 'NA'
