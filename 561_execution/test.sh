export PYTHONPATH=$PYTHONPATH:./
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

# Uncommented array represents proteins to execute 
#declare -a arr=("HSF1" "BARHL2" "NR2C2" "ZEB1" "ELK4" "NFIC" "POU2F2" "PRDM1" "RFX5" "TEAD1" "TCF3" "PAX5" "ESR1" "ZNF143" "BHLHE40" "REST" "TEAD4" "NR3C1" "TCF7L2" "ELF1" "NFE2" "NFKB1" "ETS1" "E2F1" "EBF1" "E2F4" "CEBPB" "EGR1" "ZNF263" "CTCF")
#declare -a arr=("NFIC")
declare -a arr=("HSF1")

# now loop through the above array
for i in "${arr[@]}"
do
   echo "Training a PSSM + DNA shape classifier for protein $i.";
	time python2.7 ../DNAshapedTFBS.py PSSM_ALL -f $(echo $(ls -1 PFM/"$i"/*.pfm)) \
    -i foreground/fasta/"$i".fa -I foreground/bed/"$i" \
    -b background/fasta/"$i".fa -B background/bed/"$i" \
    -o "$i" \
    -1 $helt $mgw $prot $roll -2 $helt2 $mgw2 $roll2 -n;
done
