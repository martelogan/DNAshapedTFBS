from utilities import *
from the_constants import BWTOOL, DNASHAPEINTER


def get_scores(in_file, shape=None, scaled=False):
    """ Get DNAshape values on single lines. """
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


# def combine_hits_shapes(hits, shapes, extension=0, binary_encoding=False):
#     """ Combine DNA sequence and shape features.

#     The hit scores (PSSM or TFFM) or 4-bits encoding are combined with DNAshape
#     features in vectors for classif.

#     """
#     comb = []
#     index = -1
#     numrows = len(shapes)    # 3 rows in your example
#     numcols = len(shapes[0]) # 2 columns in your example
#     # print('indx_len: ')
#     # print(numrows)
#     # print('index_len: ')
#     # print(numcols)
#     # print(shapes[7][5])
#     for hit in hits:
#         if hit:
#             index += 1
#             if shapes:
#                 hit_shapes = []
#                 #go from 0-5, skips 6, then manually add 7
#                 for indx in xrange(6):
#                     try:
#                         hit_shapes += shapes[indx][index]
#                         if (not binary_encoding and
#                             (len(hit_shapes) ==
#                                 6 * (len(hit) + 2 * extension)
#                             )
#                         ):
#                             hit_shapes += shapes[7][index]
#                             comb.append([hit.score] + hit_shapes)
#                         elif (binary_encoding and
#                             (len(hit_shapes) ==
#                                 6 * (len(hit) / 4 + 2 * extension)
#                             )
#                         ):
#                             comb.append(hit + hit_shapes)
#                     except Exception, e:
#                         continue
#             elif binary_encoding:
#                 comb.append(hit)
#             else:
#                 comb.append([hit.score])
#     return comb
def combine_hits_shapes(hits, shapes, extension=0, binary_encoding=False):
    """ Combine DNA sequence and shape features.
    The hit scores (PSSM or TFFM) or 4-bits encoding are combined with DNAshape
    features in vectors for classif.
    """
    comb = []
    index = -1
    for hit in hits:
        if hit:
            index += 1
            if shapes:
                hit_shapes = []
                for indx in xrange(len(shapes)):
                    hit_shapes += shapes[indx][index]
                if (not binary_encoding and
                        (len(hit_shapes) ==
                            len(shapes) * (len(hit) + 2 * extension)
                        )
                   ):
                    #comb.append([hit.score] + hit_shapes)
                    comb.append(hit_shapes)
                    print "Hi"
                # elif (binary_encoding and
                #         (len(hit_shapes) ==
                #             len(shapes) * (len(hit) / 4 + 2 * extension)
                #         )
                #      ):
                #     comb.append(hit + hit_shapes)
            # elif binary_encoding:
            #     comb.append(hit)
            # else:
            #     comb.append([hit.score])
    return comb


def construct_HitShapeFlex_vector(hits, shapes, flex_evals):
    """ Construct feature vector for classification """
    comb = []
    index = -1
    for hit in hits:
        if hit:
            index += 1
            if shapes:
                hit_shapes = []
                for indx in xrange(len(shapes)):
                    hit_shapes += shapes[indx][index]
                if (    (len(hit_shapes) ==
                            len(shapes) * (len(hit))
                        )
                    ):
                    # comb.append([hit.score] + hit_shapes)
                    # NOTE: flex_evals[index] is always a 1d array
                    comb.append(hit_shapes + flex_evals[index])
    return comb


def get_shapes(hits, bed_file, first_shape, second_shape, extension=0,
        scaled=False):
    """ Retrieve DNAshape feature values for the hits. """
    bigwigs = first_shape + second_shape
    print(bigwigs)
    import subprocess
    import os
    peaks_pos = get_positions_from_bed(bed_file)
    with open(os.devnull, 'w') as devnull:
        tmp_file = print_extended_hits(hits, peaks_pos, extension)
        #MODIFIED HERE TO REMOVE MGW2
        shapes = ['HelT', 'ProT', 'MGW', 'Roll', 'HelT2', 'ProT2',
                'Roll2']
        hits_shapes = []
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
                    hits_shapes.append(get_scores(out_file, shapes[indx], scaled))
                else:
                    hits_shapes.append(get_scores(out_file, shapes[indx]))
        subprocess.call(['rm', '-f', '{0}.HelT'.format(tmp_file),
            '{0}.MGW'.format(tmp_file), '{0}.ProT'.format(tmp_file),
            '{0}.Roll'.format(tmp_file), '{0}.HelT2'.format(tmp_file),
            '{0}.ProT2'.format(tmp_file), '{0}.Roll2'.format(tmp_file),tmp_file])
        
        return hits_shapes
