export PYTHONPATH=$PYTHONPATH:./
 Define where are located the DNA shape data from GBshape
hg19=/Volumes/Germanium/hg19
helt=$hg19/hg19.HelT.bigWig;
mgw=$hg19/hg19.MGW.bigWig;
prot=$hg19/hg19.ProT.bigWig;
roll=$hg19/hg19.Roll.bigWig;
helt2=$hg19/hg19.HelT.2nd.wig.bw;
mgw2=$hg19/hg19.MGW.2nd.wig.bw;
prot2=$hg19/hg19.ProT.2nd.wig.bw;
roll2=$hg19/hg19.Roll.2nd.wig.bw;


echo "Training a PSSM + DNA shape classifier.";
time python2.7 ../DNAshapedTFBS.py trainPSSM -f MA0635.1.pfm \
    -i foreground/train.fa -I foreground/train.bed \
    -b background/train.fa -B background/train.bed \
    -o DNAshapedPSSM_classifier \
    -1 $helt $mgw $prot $roll -2 $helt2 $mgw2 $prot2 $roll2 -n;

echo "Applying the trained PSSM + DNA shape classifier on foreground sequences.";
time python2.7 ../DNAshapedTFBS.py applyPSSM -f MA0635.1.pfm \
    -i foreground/test.fa -I foreground/test.bed \
    -c DNAshapedPSSM_classifier.pkl -o DNAshapedPSSM_fg_predictions.txt \
    -1 $helt $mgw $prot $roll -2 $helt2 $mgw2 $prot2 $roll2 -n;

echo "Applying the trained PSSM + DNA shape classifier on background sequences.";
time python2.7 ../DNAshapedTFBS.py applyPSSM -f MA0635.1.pfm \
    -i background/test.fa -I background/test.bed \
    -c DNAshapedPSSM_classifier.pkl -o DNAshapedPSSM_bg_predictions.txt \
    -1 $helt $mgw $prot $roll -2 $helt2 $mgw2 $prot2 $roll2 -n;

