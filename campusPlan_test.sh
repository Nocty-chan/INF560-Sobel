#!/bin/bash

make

INPUT_IMAGE=images/original/Campusplan-Hausnr.gif
OUTPUT_IMAGE=images/processed/Campusplan-Hausnr-sobel.gif


for i in 1 2 3 4 5 6 7 8 9 10 15 20; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "Running test on $i processes.\n"
    OMP_NUM_THREADS=1 srun -n $i ./sobelf $INPUT_IMAGE $OUTPUT_IMAGE

done
