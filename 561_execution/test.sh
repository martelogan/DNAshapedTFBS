export PYTHONPATH=$PYTHONPATH:./
#Define where are located the DNA shape data from GBshape
hg19=/Volumes/Germanium/hg19
helt=$hg19/hg19.HelT.bigWig;
mgw=$hg19/hg19.MGW.bigWig;
prot=$hg19/hg19.ProT.bigWig;
roll=$hg19/hg19.Roll.bigWig;
helt2=$hg19/hg19.HelT.2nd.wig.bw;
mgw2=$hg19/hg19.MGW.2nd.wig.bw;
prot2=$hg19/hg19.ProT.2nd.wig.bw;
roll2=$hg19/hg19.Roll.2nd.wig.bw;

## declare an array variable
# declare -a arr=("BARHL2" "CEBPB" "BHLHE40" "E2F1" "E2F4" "EBF1" "ELF1" "ELK4" "ETS1" "HSF1" "NFE2" "NFKB1" "NRF1" "PAX5" "POU2F2" "RFX5" "SREBF1" "STAT1" "TCF3" "TCF7L2" "TEAD1" "TEAD4" "NFIC" "EGR1" "ESR1" "GATA3" "CTCF" "NR2C2" "NR3C1" "PRDM1" "REST" "ZEB1" "ZNF143" "ZNF263")
declare -a arr=("ZNF143" "STAT1")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "Training a PSSM + DNA shape classifier for protein $i.";
	time python2.7 ../DNAshapedTFBS.py PSSM_ALL -f $(echo $(ls -1 PFM/"$i"/*.pfm)) \
    -i foreground/fasta/"$i".fa -I foreground/bed/"$i" \
    -b background/fasta/"$i".fa -B background/bed/"$i" \
    -o DNAshapedPSSM_classifier_"$i" \
    -1 $helt $mgw $prot $roll -2 $helt2 $mgw2 $roll2 -n;
done