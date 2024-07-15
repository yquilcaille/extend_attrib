#!/bin/bash 
module load conda/2022
source activate mesmer_dev

for subindex_csl in {0..72}
do
    python treatment_ERA5regridded.py $subindex_csl &
done




