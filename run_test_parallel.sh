INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

for j in 1, 2, 5, 7, 10, 15, 20; do
echo "Running test for $j processes"
for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST"
    srun -n $j ./sobelf $i $DEST

done
done
