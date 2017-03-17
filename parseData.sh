g++ -o parseData parser.cpp

INPUT_DIR=~/Téléchargements/outs
OUTPUT_DIR=results
mkdir $OUTPUT_DIR 2>/dev/null

for i in $INPUT_DIR/*.out ; do
    DEST=$OUTPUT_DIR/`basename $i .out`-processed.out
    echo "Processing  $i -> $DEST"
    ./parseData $i > $DEST

done
