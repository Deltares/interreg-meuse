#!/bin/bash
#$ -q normal-e5-c7
#$ -cwd
#$ -m bea
#$ -pe distrib 1
#$ -V
#$ -N analysis
#$ -j yes


# Initiating snakemake and running workflow in cluster mode
source /u/bouaziz/miniconda3/bin/activate hydromt-wflow

#conda config --set channel_priority strict

ROOT="/u/bouaziz/interreg-meuse/src/model_building/calibration_linux"
cd "${ROOT}"

python scripts/combine_calib_results.py

conda deactivate