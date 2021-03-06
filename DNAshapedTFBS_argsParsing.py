from DNAshapedTFBS_classification import dna_shape_and_tffm_train_classifier
from DNAshapedTFBS_classification import dna_shape_and_pssm_train_classifier
from DNAshapedTFBS_classification import dna_shape_and_binary_train_classifier
from DNAshapedTFBS_classification import custom_train_classifier
from DNAshapedTFBS_classification import dna_shape_and_tffm_apply_classifier
from DNAshapedTFBS_classification import dna_shape_and_pssm_apply_classifier
from DNAshapedTFBS_classification import dna_shape_and_binary_apply_classifier
from DNAshapedTFBS_classification import custom_apply_classifier
from DNAshapedTFBS_classification import custom_train_and_validate_classifier

# TODO: change 2nd order nargs to 4 where appropriate when we re-download MGW2


def pssm_train_args_parsing(subparsers):
    """ Train the PSSM + DNA shape classifier. """
    help_str = "Train the PSSM + DNA shape classifier."
    parser_t = subparsers.add_parser('trainPSSM', help=help_str)
    jaspar_grp = parser_t.add_mutually_exclusive_group(required=True)
    help_str = 'JASPAR ID corresponding to the TF '
    help_str += 'binding profile to be used.'
    jaspar_grp.add_argument('-j', '--jaspar', type=str, dest='jasparid',
                            action='store', help=help_str)
    help_str = 'JASPAR file containing the TF binding profile in the '
    help_str += 'JASPAR format.'
    jaspar_grp.add_argument('-f', '--jasparfile', type=str, dest='jasparfile',
                            action='store', help=help_str)
    help_str = 'Input fasta file containing the foreground sequences.'
    parser_t.add_argument('-i', '--fg_fasta', required=True, type=str,
                          dest='fg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of foreground sequences.'
    parser_t.add_argument('-I', '--fg_bed', required=True, type=str,
                          dest='fg_bed', action='store', help=help_str)
    help_str = 'Input fasta containing the background sequences.'
    parser_t.add_argument('-b', '--bg_fasta', required=True, type=str,
                          dest='bg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of background sequences.'
    parser_t.add_argument('-B', '--bg_bed', required=True, type=str,
                          dest='bg_bed', action='store', help=help_str)
    parser_t.add_argument('-o', '--outfile', required=True, type=str,
                          dest='output', action='store',
                          help='Output base name (.pkl will be added).')
    help_str = 'List of 1st-order bigWig files as\n'
    help_str += '-1 HelT ProT MGW Roll'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-1', '--1storder', required=True, type=str,
                          dest='shape_first_order', action='store', nargs=4,
                          help=help_str)
    help_str = 'List of 2nd-order bigWig files as\n'
    help_str += '-2 HelT2 ProT2 MGW2 Roll2'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-2', '--2ndorder', required=False, type=str,
                          dest='shape_second_order', action='store', nargs=4,
                          default=[], help=help_str)
    parser_t.add_argument('-n', '--scaled', required=False, dest='scaled',
                          action='store_true', default=False,
                          help='Scale DNAshape values in [0, 1]')
    help_str = 'Extension to be considered around TFBSs with DNAshapes'
    help_str += ' (default:0).'
    parser_t.add_argument('-e', '--extension', required=False, type=int,
                          dest='extension', action='store', default=0,
                          help=help_str)
    parser_t.set_defaults(func=dna_shape_and_pssm_train_classifier)


def pssm_apply_args_parsing(subparsers):
    """ Apply the PSSM + DNA shape classifier. """
    help_str = 'Apply the PSSM + DNA shape classifier.'
    parser_a = subparsers.add_parser('applyPSSM', help=help_str)
    jaspar_grp = parser_a.add_mutually_exclusive_group(required=True)
    help_str = 'JASPAR ID corresponding to the TF '
    help_str += 'binding profile to be used.'
    jaspar_grp.add_argument('-j', '--jaspar', type=str, dest='jasparid',
                            action='store', help=help_str)
    help_str = 'JASPAR file containing the TF binding profile in the '
    help_str += 'JASPAR format.'
    jaspar_grp.add_argument('-f', '--jasparfile', type=str, dest='jasparfile',
                            action='store', help=help_str)
    help_str = 'Input fasta file containing the sequences.'
    parser_a.add_argument('-i', '--input_fasta', required=True, type=str,
                          dest='in_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of the sequences.'
    parser_a.add_argument('-I', '--input_bed', required=True, type=str,
                          dest='in_bed', action='store', help=help_str)
    parser_a.add_argument('-c', '--classifier', required=True, type=str,
                          dest='classifier', action='store',
                          help='Classifier (.pkl file).')
    parser_a.add_argument('-o', '--outfile', required=True, type=str,
                          dest='output', action='store',
                          help='Output result file')
    help_str = 'List of 1st-order bigWig files as\n'
    help_str += '-1 HelT ProT MGW Roll'
    help_str += ' (note that we assume such an order)'
    parser_a.add_argument('-1', '--1storder', required=True, type=str,
                          dest='shape_first_order', action='store', nargs=4,
                          help=help_str)
    help_str = 'List of 2nd-order bigWig files as\n'
    help_str += '-2 HelT2 ProT2 MGW2 Roll2'
    help_str += ' (note that we assume such an order)'
    parser_a.add_argument('-2', '--2ndorder', required=False, type=str,
                          dest='shape_second_order', action='store', nargs=4,
                          default=[], help=help_str)
    parser_a.add_argument('-n', '--scaled', required=False, dest='scaled',
                          action='store_true', default=False,
                          help='Scale DNAshape values in [0, 1]')
    help_str = 'Extension to be considered around TFBSs with DNAshapes'
    help_str += ' (default:0).'
    parser_a.add_argument('-e', '--extension', required=False, type=int,
                          dest='extension', action='store', default=0,
                          help=help_str)
    help_str = 'Probability threshold to predict a hit (default=0.5).'
    parser_a.add_argument('-v', '--threshold', required=False,
                          dest='threshold', action='store', default=0.5,
                          type=float, help=help_str)
    parser_a.set_defaults(func=dna_shape_and_pssm_apply_classifier)


def tffm_train_args_parsing(subparsers):
    """ Train the TFFM + DNA shape classifier. """
    help_str = "Train the TFFM + DNA shape classifier."
    parser_t = subparsers.add_parser('trainTFFM', help=help_str)
    parser_t.add_argument('-T', '--tffmfile', required=True, dest='tffm_file',
                          action='store', type=str, help='TFFM XML file.')
    parser_t.add_argument('-t', '--tffm_kind', required=False,
                          dest='tffm_kind', action='store', type=str,
                          choices=['first_order', 'detailed'],
                          default='first_order',
                          help='TFFM kind ("first_order" or "detailed").')
    help_str = 'Input fasta file containing the foreground sequences.'
    parser_t.add_argument('-i', '--fg_fasta', required=True, type=str,
                          dest='fg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of foreground sequences.'
    parser_t.add_argument('-I', '--fg_bed', required=True, type=str,
                          dest='fg_bed', action='store', help=help_str)
    help_str = 'Input fasta containing the background sequences.'
    parser_t.add_argument('-b', '--bg_fasta', required=True, type=str,
                          dest='bg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of background sequences.'
    parser_t.add_argument('-B', '--bg_bed', required=True, type=str,
                          dest='bg_bed', action='store', help=help_str)
    parser_t.add_argument('-o', '--outfile', required=True, type=str,
                          dest='output', action='store',
                          help='Output base name (.pkl will be added).')
    help_str = 'List of 1st-order bigWig files as\n'
    help_str += '-1 HelT ProT MGW Roll'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-1', '--1storder', required=True, type=str,
                          dest='shape_first_order', action='store', nargs=4,
                          help=help_str)
    help_str = 'List of 2nd-order bigWig files as\n'
    help_str += '-2 HelT2 ProT2 MGW2 Roll2'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-2', '--2ndorder', required=False, type=str,
                          dest='shape_second_order', action='store', nargs=4,
                          default=[], help=help_str)
    parser_t.add_argument('-n', '--scaled', required=False, dest='scaled',
                          action='store_true', default=False,
                          help='Scale DNAshape values in [0, 1]')
    help_str = 'Extension to be considered around TFBSs with DNAshapes'
    help_str += ' (default:0).'
    parser_t.add_argument('-e', '--extension', required=False, type=int,
                          dest='extension', action='store', default=0,
                          help=help_str)
    parser_t.set_defaults(func=dna_shape_and_tffm_train_classifier)


def tffm_apply_args_parsing(subparsers):
    """ Apply the TFFM + DNA shape classifier. """
    help_str = 'Apply the TFFM + DNA shape classifier.'
    parser_a = subparsers.add_parser('applyTFFM', help=help_str)
    parser_a.add_argument('-T', '--tffmfile', required=True, dest='tffm_file',
                          action='store', type=str, help='TFFM XML file.')
    parser_a.add_argument('-t', '--tffm_kind', required=False,
                          dest='tffm_kind', action='store', type=str,
                          choices=['first_order', 'detailed'],
                          default='first_order',
                          help='TFFM kind ("first_order" or "detailed").')
    help_str = 'Input fasta file containing the sequences.'
    parser_a.add_argument('-i', '--input_fasta', required=True, type=str,
                          dest='in_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of the sequences.'
    parser_a.add_argument('-I', '--input_bed', required=True, type=str,
                          dest='in_bed', action='store', help=help_str)
    parser_a.add_argument('-c', '--classifier', required=True, type=str,
                          dest='classifier', action='store',
                          help='Classifier (.pkl file).')
    parser_a.add_argument('-o', '--outfile', required=True, type=str,
                          dest='output', action='store',
                          help='Output result file')
    help_str = 'List of 1st-order bigWig files as\n'
    help_str += '-1 HelT ProT MGW Roll'
    help_str += ' (note that we assume such an order)'
    parser_a.add_argument('-1', '--1storder', required=True, type=str,
                          dest='shape_first_order', action='store', nargs=4,
                          help=help_str)
    help_str = 'List of 2nd-order bigWig files as: '
    help_str += '-2 HelT2 ProT2 MGW2 Roll2'
    help_str += ' (note that we assume such an order)'
    parser_a.add_argument('-2', '--2ndorder', required=False, type=str,
                          dest='shape_second_order', action='store', nargs=4,
                          default=[], help=help_str)
    parser_a.add_argument('-n', '--scaled', required=False, dest='scaled',
                          action='store_true', default=False,
                          help='Scale DNAshape values in [0, 1]')
    help_str = 'Extension to be considered around TFBSs with DNAshapes'
    help_str += ' (default:0).'
    parser_a.add_argument('-e', '--extension', required=False, type=int,
                          dest='extension', action='store', default=0,
                          help=help_str)
    help_str = 'Probability threshold to predict a hit (default=0.5).'
    parser_a.add_argument('-v', '--threshold', required=False,
                          dest='threshold', action='store', default=0.5,
                          type=float, help=help_str)
    parser_a.set_defaults(func=dna_shape_and_tffm_apply_classifier)


def binary_train_args_parsing(subparsers):
    """ Train the 4-bits + DNA shape classifier. """
    help_str = "Train the 4-bits + DNA shape classifier."
    parser_t = subparsers.add_parser('train4bits', help=help_str)
    jaspar_grp = parser_t.add_mutually_exclusive_group(required=True)
    help_str = 'JASPAR ID corresponding to the TF '
    help_str += 'binding profile to be used.'
    jaspar_grp.add_argument('-j', '--jaspar', type=str, dest='jasparid',
                            action='store', help=help_str)
    help_str = 'JASPAR file containing the TF binding profile in the '
    help_str += 'JASPAR format.'
    jaspar_grp.add_argument('-f', '--jasparfile', type=str, dest='jasparfile',
                            action='store', help=help_str)
    help_str = 'Input fasta file containing the foreground sequences.'
    parser_t.add_argument('-i', '--fg_fasta', required=True, type=str,
                          dest='fg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of foreground sequences.'
    parser_t.add_argument('-I', '--fg_bed', required=True, type=str,
                          dest='fg_bed', action='store', help=help_str)
    help_str = 'Input fasta containing the background sequences.'
    parser_t.add_argument('-b', '--bg_fasta', required=True, type=str,
                          dest='bg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of background sequences.'
    parser_t.add_argument('-B', '--bg_bed', required=True, type=str,
                          dest='bg_bed', action='store', help=help_str)
    parser_t.add_argument('-o', '--outfile', required=True, type=str,
                          dest='output', action='store',
                          help='Output base name (.pkl will be added).')
    help_str = 'List of 1st-order bigWig files as\n'
    help_str += '-1 HelT ProT MGW Roll'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-1', '--1storder', required=True, type=str,
                          dest='shape_first_order', action='store', nargs=4,
                          help=help_str)
    help_str = 'List of 2nd-order bigWig files as\n'
    help_str += '-2 HelT2 ProT2 MGW2 Roll2'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-2', '--2ndorder', required=False, type=str,
                          dest='shape_second_order', action='store', nargs=4,
                          default=[], help=help_str)
    parser_t.add_argument('-n', '--scaled', required=False, dest='scaled',
                          action='store_true', default=False,
                          help='Scale DNAshape values in [0, 1]')
    help_str = 'Extension to be considered around TFBSs with DNAshapes'
    help_str += ' (default:0).'
    parser_t.add_argument('-e', '--extension', required=False, type=int,
                          dest='extension', action='store', default=0,
                          help=help_str)
    parser_t.set_defaults(func=dna_shape_and_binary_train_classifier)


def binary_apply_args_parsing(subparsers):
    """ Apply the 4-bits + DNA shape classifier. """
    help_str = 'Apply the 4-bits + DNA shape classifier.'
    parser_a = subparsers.add_parser('apply4bits', help=help_str)
    jaspar_grp = parser_a.add_mutually_exclusive_group(required=True)
    help_str = 'JASPAR ID corresponding to the TF '
    help_str += 'binding profile to be used.'
    jaspar_grp.add_argument('-j', '--jaspar', type=str, dest='jasparid',
                            action='store', help=help_str)
    help_str = 'JASPAR file containing the TF binding profile in the '
    help_str += 'JASPAR format.'
    jaspar_grp.add_argument('-f', '--jasparfile', type=str, dest='jasparfile',
                            action='store', help=help_str)
    help_str = 'Input fasta file containing the sequences.'
    parser_a.add_argument('-i', '--input_fasta', required=True, type=str,
                          dest='in_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of the sequences.'
    parser_a.add_argument('-I', '--input_bed', required=True, type=str,
                          dest='in_bed', action='store', help=help_str)
    parser_a.add_argument('-c', '--classifier', required=True, type=str,
                          dest='classifier', action='store',
                          help='Classifier (.pkl file).')
    parser_a.add_argument('-o', '--outfile', required=True, type=str,
                          dest='output', action='store',
                          help='Output result file')
    help_str = 'List of 1st-order bigWig files as\n'
    help_str += '-1 HelT ProT MGW Roll'
    help_str += ' (note that we assume such an order)'
    parser_a.add_argument('-1', '--1storder', required=True, type=str,
                          dest='shape_first_order', action='store', nargs=4,
                          help=help_str)
    help_str = 'List of 2nd-order bigWig files as\n'
    help_str += '-2 HelT2 ProT2 MGW2 Roll2'
    help_str += ' (note that we assume such an order)'
    parser_a.add_argument('-2', '--2ndorder', required=False, type=str,
                          dest='shape_second_order', action='store', nargs=4,
                          default=[], help=help_str)
    parser_a.add_argument('-n', '--scaled', required=False, dest='scaled',
                          action='store_true', default=False,
                          help='Scale DNAshape values in [0, 1]')
    help_str = 'Extension to be considered around TFBSs with DNAshapes'
    help_str += ' (default:0).'
    parser_a.add_argument('-e', '--extension', required=False, type=int,
                          dest='extension', action='store', default=0,
                          help=help_str)
    help_str = 'Probability threshold to predict a hit (default=0.5).'
    parser_a.add_argument('-v', '--threshold', required=False,
                          dest='threshold', action='store', default=0.5,
                          type=float, help=help_str)
    parser_a.set_defaults(func=dna_shape_and_binary_apply_classifier)


def custom_train_args_parsing(subparsers):
    """ Train a custom classifier (based on available combinations). """
    help_str = "Train a custom classifier."
    parser_t = subparsers.add_parser('custom_train', help=help_str)
    jaspar_grp = parser_t.add_mutually_exclusive_group(required=True)
    help_str = 'JASPAR ID corresponding to the TF '
    help_str += 'binding profile to be used.'
    jaspar_grp.add_argument('-j', '--jaspar', type=str, dest='jasparid',
                            action='store', help=help_str)
    help_str = 'JASPAR file containing the TF binding profile in the '
    help_str += 'JASPAR format.'
    jaspar_grp.add_argument('-f', '--jasparfile', type=str, dest='jasparfile',
                            action='store', help=help_str)
    help_str = 'Input fasta file containing the foreground sequences.'
    parser_t.add_argument('-i', '--fg_fasta', required=True, type=str,
                          dest='fg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of foreground sequences.'
    parser_t.add_argument('-I', '--fg_bed', required=True, type=str,
                          dest='fg_bed', action='store', help=help_str)
    help_str = 'Input fasta containing the background sequences.'
    parser_t.add_argument('-b', '--bg_fasta', required=True, type=str,
                          dest='bg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of background sequences.'
    parser_t.add_argument('-B', '--bg_bed', required=True, type=str,
                          dest='bg_bed', action='store', help=help_str)
    parser_t.add_argument('-o', '--outfile', required=True, type=str,
                          dest='output', action='store',
                          help='Output base name (.pkl will be added).')
    help_str = 'List of 1st-order bigWig files as\n'
    help_str += '-1 HelT ProT MGW Roll'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-1', '--1storder', required=True, type=str,
                          dest='shape_first_order', action='store', nargs=4,
                          help=help_str)
    help_str = 'List of 2nd-order bigWig files as\n'
    help_str += '-2 HelT2 ProT2 Roll2'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-2', '--2ndorder', required=False, type=str,
                          dest='shape_second_order', action='store', nargs=3,
                          default=[], help=help_str)
    parser_t.add_argument('-n', '--scaled', required=False, dest='scaled',
                          action='store_true', default=False,
                          help='Scale DNAshape values in [0, 1]')
    help_str = 'Extension to be considered around TFBSs with DNAshapes'
    help_str += ' (default:0).'
    parser_t.add_argument('-e', '--extension', required=False, type=int,
                          dest='extension', action='store', default=0,
                          help=help_str)
    parser_t.set_defaults(func=custom_train_and_validate_classifier)
    help_str = 'Type of feature vector to construct\n'
    help_str += '(0 = seq_and_dna_shape| 1 = dna_shape_only | 2 = dna_shape_and_flexibility)'
    help_str += ' (default : 0).'
    parser_t.add_argument('-q', '--feature_vector_type', required=False, type=int,
                          dest='feature_vector_type', action='store', default=0,
                          help=help_str)
    help_str = 'Type of DNA sequence feature to consider\n'
    help_str += '(0 = PSSM | 1 = TFFM | 2 = 4bit)'
    help_str += ' (default : 0).'
    parser_t.add_argument('-s', '--seq_feature', required=False, type=int,
                          dest='seq_feature', action='store', default=0,
                          help=help_str)
    help_str = 'Are we evaluating flexibility with a wrapper function?\n'
    help_str += '(0 = False | 1 = True)'
    help_str += ' (default : 0).'
    parser_t.add_argument('-E', '--is_eval_f', required=False, type=int,
                          dest='is_eval_f', action='store', default=0,
                          help=help_str)
    parser_t.set_defaults(func=custom_train_classifier)


def custom_apply_args_parsing(subparsers):
    """ Apply a custom classifier (based on available combinations). """
    help_str = "Apply a custom classifier."
    parser_a = subparsers.add_parser('custom_apply', help=help_str)
    jaspar_grp = parser_a.add_mutually_exclusive_group(required=True)
    help_str = 'JASPAR ID corresponding to the TF '
    help_str += 'binding profile to be used.'
    jaspar_grp.add_argument('-j', '--jaspar', type=str, dest='jasparid',
                            action='store', help=help_str)
    help_str = 'JASPAR file containing the TF binding profile in the '
    help_str += 'JASPAR format.'
    jaspar_grp.add_argument('-f', '--jasparfile', type=str, dest='jasparfile',
                            action='store', help=help_str)
    help_str = 'Input fasta file containing the sequences.'
    parser_a.add_argument('-i', '--input_fasta', required=True, type=str,
                          dest='in_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of the sequences.'
    parser_a.add_argument('-I', '--input_bed', required=True, type=str,
                          dest='in_bed', action='store', help=help_str)
    parser_a.add_argument('-c', '--classifier', required=True, type=str,
                          dest='classifier', action='store',
                          help='Classifier (.pkl file).')
    parser_a.add_argument('-o', '--outfile', required=True, type=str,
                          dest='output', action='store',
                          help='Output result file')
    help_str = 'List of 1st-order bigWig files as\n'
    help_str += '-1 HelT ProT MGW Roll'
    help_str += ' (note that we assume such an order)'
    parser_a.add_argument('-1', '--1storder', required=True, type=str,
                          dest='shape_first_order', action='store', nargs=4,
                          help=help_str)
    help_str = 'List of 2nd-order bigWig files as\n'
    help_str += '-2 HelT2 ProT2 MGW2 Roll2'
    help_str += ' (note that we assume such an order)'
    parser_a.add_argument('-2', '--2ndorder', required=False, type=str,
                          dest='shape_second_order', action='store', nargs=3,
                          default=[], help=help_str)
    parser_a.add_argument('-n', '--scaled', required=False, dest='scaled',
                          action='store_true', default=False,
                          help='Scale DNAshape values in [0, 1]')
    help_str = 'Extension to be considered around TFBSs with DNAshapes'
    help_str += ' (default:0).'
    parser_a.add_argument('-e', '--extension', required=False, type=int,
                          dest='extension', action='store', default=0,
                          help=help_str)
    help_str = 'Probability threshold to predict a hit (default=0.5).'
    parser_a.add_argument('-v', '--threshold', required=False,
                          dest='threshold', action='store', default=0.5,
                          type=float, help=help_str)
    help_str = 'Type of feature vector to construct\n'
    help_str += '(0 = seq_and_dna_shape| 1 = dna_shape_only | 2 = dna_shape_and_flexibility)'
    help_str += ' (default : 0).'
    parser_a.add_argument('-q', '--feature_vector_type', required=False, type=int,
                          dest='feature_vector_type', action='store', default=0,
                          help=help_str)
    help_str = 'Type of DNA sequence feature to consider\n'
    help_str += '(0 = PSSM | 1 = TFFM | 2 = 4bit)'
    help_str += ' (default : 0).'
    parser_a.add_argument('-s', '--seq_feature', required=False, type=int,
                          dest='seq_feature', action='store', default=0,
                          help=help_str)
    help_str = 'Are we evaluating flexibility with a wrapper function?\n'
    help_str += '(0 = False | 1 = True)'
    help_str += ' (default : 0).'
    parser_a.add_argument('-E', '--is_eval_f', required=False, type=int,
                          dest='is_eval_f', action='store', default=0,
                          help=help_str)
    parser_a.set_defaults(func=custom_apply_classifier)


def custom_train_and_validate_args_parsing(subparsers):
    """ Train and apply the PSSM + DNA shape classifier. """
    help_str = "Train and apply the PSSM + DNA shape classifier."
    parser_t = subparsers.add_parser('custom_train_and_apply', help=help_str)
    jaspar_grp = parser_t.add_mutually_exclusive_group(required=True)
    help_str = 'JASPAR ID corresponding to the TF '
    help_str += 'binding profile to be used.'
    jaspar_grp.add_argument('-j', '--jaspar', type=str, dest='jasparid',
                            action='store', help=help_str)
    help_str = 'JASPAR file containing the TF binding profile in the '
    help_str += 'JASPAR format.'
    jaspar_grp.add_argument('-f', '--jasparfile', type=str, dest='jasparfile',
                            action='store', help=help_str)
    help_str = 'Input fasta file containing the foreground sequences.'
    parser_t.add_argument('-i', '--fg_fasta', required=True, type=str,
                          dest='fg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of foreground sequences.'
    parser_t.add_argument('-I', '--fg_bed', required=True, type=str,
                          dest='fg_bed', action='store', help=help_str)
    help_str = 'Input fasta containing the background sequences.'
    parser_t.add_argument('-b', '--bg_fasta', required=True, type=str,
                          dest='bg_fasta', action='store', help=help_str)
    help_str = 'Input bed file w/ positions of background sequences.'
    parser_t.add_argument('-B', '--bg_bed', required=True, type=str,
                          dest='bg_bed', action='store', help=help_str)
    parser_t.add_argument('-o', '--outfile', required=True, type=str,
                          dest='output', action='store',
                          help='Output base name (.pkl will be added).')
    help_str = 'List of 1st-order bigWig files as\n'
    help_str += '-1 HelT ProT MGW Roll'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-1', '--1storder', required=True, type=str,
                          dest='shape_first_order', action='store', nargs=4,
                          help=help_str)
    help_str = 'List of 2nd-order bigWig files as\n'
    help_str += '-2 HelT2 ProT2 Roll2'
    help_str += ' (note that we assume such an order)'
    parser_t.add_argument('-2', '--2ndorder', required=False, type=str,
                          dest='shape_second_order', action='store', nargs=3,
                          default=[], help=help_str)
    parser_t.add_argument('-n', '--scaled', required=False, dest='scaled',
                          action='store_true', default=False,
                          help='Scale DNAshape values in [0, 1]')
    help_str = 'Extension to be considered around TFBSs with DNAshapes'
    help_str += ' (default:0).'
    parser_t.add_argument('-e', '--extension', required=False, type=int,
                          dest='extension', action='store', default=0,
                          help=help_str)
    parser_t.set_defaults(func=custom_train_and_validate_classifier)
    help_str = 'Type of feature vector to construct\n'
    help_str += '(0 = seq_and_dna_shape| 1 = dna_shape_only | 2 = dna_shape_and_flexibility)'
    help_str += ' (default : 0).'
    parser_t.add_argument('-q', '--feature_vector_type', required=False, type=int,
                          dest='feature_vector_type', action='store', default=0,
                          help=help_str)
    help_str = 'Type of DNA sequence feature to consider\n'
    help_str += '(0 = PSSM | 1 = TFFM | 2 = 4bit)'
    help_str += ' (default : 0).'
    parser_t.add_argument('-s', '--seq_feature', required=False, type=int,
                          dest='seq_feature', action='store', default=0,
                          help=help_str)
    help_str = 'Are we evaluating flexibility with a wrapper function?\n'
    help_str += '(0 = False | 1 = True)'
    help_str += ' (default : 0).'
    parser_t.add_argument('-E', '--is_eval_f', required=False, type=bool,
                          dest='is_eval_f', action='store', default=False,
                          help=help_str)
    help_str = 'Optional experiment name (prepended to cumulative csvs)'
    parser_t.add_argument('-N', '--exp_name', required=False, type=str,
                          dest='exp_name', action='store', default='',
                          help=help_str)
    parser_t.set_defaults(func=custom_train_and_validate_classifier)
    help_str = 'Optional background name (prepended to cumulative csvs)'
    parser_t.add_argument('-Q', '--back_type', required=False, type=str,
                          dest='background_type', action='store', default='',
                          help=help_str)
    help_str = 'Optional protein name (-o value will be used if not specified)'
    parser_t.add_argument('-P', '--protein', required=False, type=str,
                          dest='protein', action='store', default='',
                          help=help_str)
    parser_t.set_defaults(func=custom_train_and_validate_classifier)


def arg_parsing():
    """ Parse the subcommand along with its arguments. """

    descr = '''
    Train or apply the DNAshape-based classifiers to a set of fasta sequences.
    '''
    import argparse
    parser = argparse.ArgumentParser(
        description=descr,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(
        help='Train or apply a PSSM/TFFM/4-bits + DNA shape classifier',
        title='Subcommands', description='Valid subcommands')
    tffm_train_args_parsing(subparsers)
    tffm_apply_args_parsing(subparsers)
    pssm_train_args_parsing(subparsers)
    pssm_apply_args_parsing(subparsers)
    binary_train_args_parsing(subparsers)
    binary_apply_args_parsing(subparsers)
    custom_train_args_parsing(subparsers)
    custom_apply_args_parsing(subparsers)
    custom_train_and_validate_args_parsing(subparsers)
    argu = parser.parse_args()
    return argu
