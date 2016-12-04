## declare an array variable
declare -a arr=("BARHL2" "CEBPB" "BHLHE40" "E2F1" "E2F4" "EBF1" "ELF1" "ELK4" "ETS1" "HSF1" "NFE2" "NFKB1" "NRF1" "PAX5" "POU2F2" "RFX5" "SREBF1" "STAT1" "TCF3" "TCF7L2" "TEAD1" "TEAD4" "NFIC" "EGR1" "ESR1" "GATA3" "CTCF" "NR2C2" "NR3C1" "PRDM1" "REST" "ZEB1" "ZNF143" "ZNF263")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "Training a PSSM + DNA shape classifier for protein $i.";
	time python2.7 ../DNAshapedTFBS.py trainAndApplyPSSM -f PFM/"$i"/$(echo $(ls -1 PFM/"$i"/*.pfm)) \
    -i foreground/fasta/"$i".fa -I foreground/bed/"$i" \
    -b background/fasta/"$i".fa -B background/bed/"$i" \
    -o DNAshapedPSSM_classifier_"$i" \
    -1 $helt $mgw $prot $roll -2 $helt2 $mgw2 $roll2 -n;
done