import os
import sys
import getopt
import csv


def main(argv):
    # GET THE INPUT AND OUTPUT FILES
    input_file = ''
    output_file = ''
    is_flex_eval = False

    try:
        opts, args = getopt.getopt(argv,"hi:o:p:f:e",["ifile=","ofile=","--protein","--family_name","is_flex_eval="])
    except getopt.GetoptError:
        print 'test.py -i <input_file> -o <output_file> -p <protein> -f <family_name> [-e]'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt in ("-p", "--protein"):
            protein_name = arg
        elif opt in ("-f", "--family_name"):
            family_name = arg
        elif opt in ("-e", "--is_flex_eval"):
            is_flex_eval = True

    if not os.path.isfile(output_file):
        with open(r'' + output_file, 'w') as output_fp:
            writer = csv.writer(output_fp)
            title_headers = [
                'TF_Family',
                'Protein',
                'Interaction_Type',
                'Significance_Threshold',
                'H_Stats_Count_Above_Threshold',
                'H_Stats_Sum_Above_Threshold',
                'H_Stats_Avg_Above_Threshold',
            ]
            writer.writerow(title_headers)

    # PERFORM THE LOGIC
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    num_thresholds = len(thresholds)

    shape_shape_sums = [0.0] * num_thresholds
    flex_flex_sums = [0.0] * num_thresholds
    shape_flex_sums = [0.0] * num_thresholds

    shape_shape_counts = [0] * num_thresholds
    flex_flex_counts = [0] * num_thresholds
    shape_flex_counts = [0] * num_thresholds

    with open(input_file, 'rb') as input_fp:
        reader = csv.reader(input_fp, delimiter=',')
        for row in reader:
            len_feature_1 = len(row[0])
            len_feature_2 = len(row[1])
            h_stat_str = row[2]
            if repr(h_stat_str) == 'nan':
                continue
            h_stat = float(h_stat_str)

            if is_flex_eval:
                feature_1_is_flex = len_feature_1 == 3
                feature_2_is_flex = len_feature_2 == 3
            else:
                feature_1_is_flex = len_feature_1 == 18
                feature_2_is_flex = len_feature_2 == 18

            has_flex_feature = (feature_1_is_flex or feature_2_is_flex)
            is_flex_flex_pair = feature_1_is_flex and feature_2_is_flex
            is_shape_shape_pair = not has_flex_feature

            for threshold_index in xrange(num_thresholds):
                threshold = thresholds[threshold_index]
                if(h_stat >= threshold):
                    if is_shape_shape_pair:
                        shape_shape_sums[threshold_index] += h_stat
                        shape_shape_counts[threshold_index] += 1
                    elif is_flex_flex_pair:
                        flex_flex_sums[threshold_index] += h_stat
                        flex_flex_counts[threshold_index] += 1
                    else:
                        shape_flex_sums[threshold_index] += h_stat
                        shape_flex_counts[threshold_index] += 1

        with open(output_file, "a") as output_fp:
            writer = csv.writer(output_fp)
            for threshold_index in xrange(num_thresholds):
                shape_shape_count = shape_shape_counts[threshold_index]
                shape_shape_sum = shape_shape_sums[threshold_index]
                shape_shape_avg = 0.0

                flex_flex_count = flex_flex_counts[threshold_index]
                flex_flex_sum = flex_flex_sums[threshold_index]
                flex_flex_avg = 0.0

                shape_flex_count = shape_flex_counts[threshold_index]
                shape_flex_sum = shape_flex_sums[threshold_index]
                shape_flex_avg = 0.0

                if shape_shape_count > 0:
                    shape_shape_avg = shape_shape_sum / shape_shape_count
                if flex_flex_count > 0:
                    flex_flex_avg = flex_flex_sum / flex_flex_count
                if shape_flex_count > 0:
                    shape_flex_avg = shape_flex_sum / shape_flex_count

                writer.writerow([family_name, protein_name, 'Shape-Shape', threshold, shape_shape_count, shape_shape_sum, shape_shape_avg])
                writer.writerow([family_name, protein_name, 'Flex-Flex', threshold, flex_flex_count, flex_flex_sum, flex_flex_avg])
                writer.writerow([family_name, protein_name, 'Shape-Flex', threshold, shape_flex_count, shape_flex_sum, shape_flex_avg])


if __name__ == "__main__":
    main(sys.argv[1:])
