#!/bin/bash 
module load conda/2024
source activate mesmer_dev

# Number of jobs to run: warning, represent about 60-70% of CPU for training; decreased from 20 to 15 because of RAM hitting max.
nb_csl=30

# running them all
#for ((index_csl=1; index_csl<=nb_csl; index_csl++));# for index_csl in {0..$nb_csl} --> doesnt work when nb_csl is a parameter
for index_csl in {0..29}
do
    python attribution_majors.py $nb_csl $index_csl &
done




