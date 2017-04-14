#!/usr/bin/env bash
# TODO: parametrize paths to config file and use python path environment for execution
export PYTHONPATH=$PYTHONPATH:./
EXPERIMENTS_PATH=experiments

# TODO: parametrize arrays as well to config file

# EXPERIMENTS TO AGGREGATE (COMBINATIONS = EXPERIMENT_TYPES * BACKGROUND_TYPES)

declare -a experiment_types=("pssm_dna_shape_and_youngs_mod")

declare -a background_types=("dinucleotide_shuffled_GM12878")

# PROTEINS TO EXECUTE (uses uncommented)

declare -a protein_names=("ZEB1" "PRDM1" "ZNF143" "ELK4" "ELF1" "PU1")

declare -a associated_family_name=("Zinc_Finger" "Zinc_Finger" "Zinc_Finger" "ETS_Family" "ETS_Family" "ETS_Family")

protein_index=$((0))
for protein_name in "${protein_names[@]}"
do
    for experiment_type in "${experiment_types[@]}"
    do
        for background_type in "${background_types[@]}"
        do
           experiment_name=exp_"$experiment_type"_"$background_type"
           echo "Aggregating interaction test for experiment $experiment_name on protein $protein_name...";
            time python2.7 compute_average.py \
            -i "$EXPERIMENTS_PATH"/"$experiment_name"/output/"$protein_name"/"$protein_name"_test_interaction_stats.csv \
            --protein "$protein_name" \
            --family_name ${associated_family_name[$protein_index]} \
            --is_flex_eval \
            -o "$experiment_name"_AGGREGATED_INTERACTION_STATS.csv;
        done
    done
    protein_index=$((experiment_type_index+1))
done
