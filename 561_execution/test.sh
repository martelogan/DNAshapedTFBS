#!/usr/bin/env bash
# TODO: parametrize paths to config file and use python path environment for execution
export PYTHONPATH=$PYTHONPATH:./
EXPERIMENTS_PATH=experiments
# MODIFY BELOW PATH TO LOCAL 'hg19' DATA DIRECTORY
hg19=/media/logan/Positron/TFBS_Research/hg19;
helt=$hg19/hg19.HelT.bigWig;
mgw=$hg19/hg19.MGW.bigWig;
prot=$hg19/hg19.ProT.bigWig;
roll=$hg19/hg19.Roll.bigWig;
helt2=$hg19/hg19.HelT.2nd.wig.bw;
mgw2=$hg19/hg19.MGW.2nd.wig.bw;
prot2=$hg19/hg19.ProT.2nd.wig.bw;
roll2=$hg19/hg19.Roll.2nd.wig.bw;

# TODO: parametrize arrays as well to config file

# EXPERIMENTS TO EXECUTE (COMBINATIONS = EXPERIMENT_TYPES * BACKGROUND_TYPES)

declare -a experiment_types=("pssm_dna_shape_only" "pssm_flex_only" "pssm_dna_shape_and_flex")
# mapped experiment_type index -> feature_vector_constant (referencing DNAshapedTFBS_constants.py)
declare -a associated_feature_vector_constants=(0 1 2)
# mapped experiment_type index -> seq_feature_constant (referencing DNAshapedTFBS_constants.py)
declare -a associated_seq_feature_type_constants=(0 0 0)

declare -a background_types=("random_hg19" "matched_percent_gc_GM12878" "dinucleotide_shuffled_GM12878" "matched_percent_gc_K562" "uniform_percent_gc_GM12878")

# Proteins to execute (uses uncommented)
#declare -a proteins=("HSF1" "BARHL2" "NR2C2" "ZEB1" "ELK4" "NFIC" "POU2F2" "PRDM1" "RFX5" "TEAD1" "TCF3" "PAX5" "ESR1" "ZNF143" "BHLHE40" "REST" "TEAD4" "NR3C1" "TCF7L2" "ELF1" "NFE2" "NFKB1" "ETS1" "E2F1" "EBF1" "E2F4" "CEBPB" "EGR1" "ZNF263" "CTCF")
#declare -a proteins=("NFIC")
declare -a protein_names=("HSF1")

experiment_type_index=$((0))
for experiment_type in "${experiment_types[@]}"
do
    for background_type in "${background_types[@]}"
    do
        for protein_name in "${protein_names[@]}"
        do
           experiment_name=exp_"$experiment_type"_"$background_type"
           echo "Creating output directory for experiment $experiment_name on protein $protein_name...";
           mkdir -p "$EXPERIMENTS_PATH"/"$experiment_name"/output/"$protein_name"/
           # IN-MEMORY TRAIN AND TEST (currently applies kfold by default)
           echo "Training and validating a classifier for experiment $experiment_name on protein $protein_name.";
            time python2.7 ../DNAshapedTFBS_classification.py custom_train_and_apply -f $(echo $(ls -1 PFMs/"$protein_name"/*.pfm)) \
            -i "$EXPERIMENTS_PATH"/"$experiment_name"/foreground/fasta/"$protein_name".fa \
            -I "$EXPERIMENTS_PATH"/"$experiment_name"/foreground/bed/"$protein_name" \
            -b "$EXPERIMENTS_PATH"/"$experiment_name"/background/fasta/"$protein_name".fa \
            -B "$EXPERIMENTS_PATH"/"$experiment_name"/background/bed/"$protein_name" \
            -1 $helt $mgw $prot $roll -2 $helt2 $mgw2 $prot2 $roll2 -n \
            --exp_name "$experiment_type" \
            --protein "$protein_name" \
            --background_type "$background_type" \
            --feature_vector_type ${associated_feature_vector_constants[${experiment_type_index}]} \
            --seq_feature_type ${associated_seq_feature_type_constants[${experiment_type_index}]} \
            -o "$EXPERIMENTS_PATH"/"$experiment_name"/output/"$protein_name"/"$protein_name";

            # INDEPENDENT TRAINING + TESTING EXECUTIONS
#            echo "Training a classifier for experiment $experiment_name on protein $protein_name.";
#            time python2.7 ../DNAshapedTFBS_classification.py custom_train -f $(echo $(ls -1 PFMs/"$i"/*.pfm)) \
#            -i "$EXPERIMENTS_PATH"/"$experiment_name"/foreground/fasta/"$protein_name".fa \
#            -I "$EXPERIMENTS_PATH"/"$experiment_name"/foreground/bed/"$protein_name" \
#            -b "$EXPERIMENTS_PATH"/"$experiment_name"/background/fasta/"$protein_name".fa \
#            -B "$EXPERIMENTS_PATH"/"$experiment_name"/background/bed/"$protein_name" \
#            -1 $helt $mgw $prot $roll -2 $helt2 $mgw2 $prot2 $roll2 -n \
#            --exp_name "$experiment_type" \
#            --protein "$protein_name" \
#            --background_type "$background_type" \
#            --feature_vector_type ${associated_feature_vector_constants[$experiment_type_index]} \
#            --seq_feature_type ${associated_seq_feature_type_constants[$experiment_type_index]} \
#            -o "$EXPERIMENTS_PATH"/"$experiment_name"/output/"$protein_name"/"$protein_name";
#
#            echo "Applying the classifier from experiment $experiment_name on protein $protein_name.";
#            time python2.7 ../DNAshapedTFBS_classification.py custom_apply -f $(echo $(ls -1 PFMs/"$i"/*.pfm)) \
#            -i "$EXPERIMENTS_PATH"/"$experiment_name"/foreground/fasta/"$protein_name".fa \
#            -I "$EXPERIMENTS_PATH"/"$experiment_name"/foreground/bed/"$protein_name" \
#            -c "$EXPERIMENTS_PATH"/"$experiment_name"/output/"$protein_name"/"$protein_name".pkl \
#            -1 $helt $mgw $prot $roll -2 $helt2 $mgw2 $prot2 $roll2 -n \
#            --exp_name "$experiment_type" \
#            --protein "$protein_name" \
#            --background_type "$background_type" \
#            --feature_vector_type ${associated_feature_vector_constants[$experiment_type_index]} \
#            --seq_feature_type ${associated_seq_feature_type_constants[$experiment_type_index]} \
#            -o "$EXPERIMENTS_PATH"/"$experiment_name"/output/"$protein_name"/"$protein_name";
        done
    done
    experiment_type_index=$((experiment_type_index+1))
done
