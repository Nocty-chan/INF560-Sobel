#!/bin/bash

make

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

for i in $INPUT_DIR/*gif ; do
    for j in 1 2 3 4 5 6 7 8 9 10 15; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST with $j threads and $1 node"
    OMP_NUM_THREADS=$j srun -n $1 ./sobelf $i $DEST 
    done
done
