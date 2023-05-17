#!/bin/bash
#$ -q normal-e3-c7
#$ -cwd
#$ -m bea
#$ -pe distrib 1
#$ -V
#$ -N snake_wflow_int
#$ -j yes


# Initiating snakemake and running workflow in cluster mode
source /u/bouaziz/miniconda3/bin/activate hydromt-wflow

#conda config --set channel_priority strict

ROOT = "/u/bouaziz/interreg-meuse/src/model_building/calibration_linux"
cd "${ROOT}"

snakemake -s Snakefile --configfile config/snake_config_model.yml --dag | dot -Tpng > dag_all.png

snakemake --unlock -s Snakefile --configfile config/snake_config_model.yml
snakemake all -s Snakefile --configfile config/snake_config_model.yml --jobs 20 --latency-wait 60 --wait-for-files --rerun-incomplete --cluster "$SNAKE_SUBMIT_JOB_INTERREG" --directory $PWD


#snakemake all -n -s Snakefile --configfile config/snake_config_model.yml -q --rerun-incomplete

# --resources mem_mb=70000
# see dry run (-n) for number of jobs. -q (quiet -- summary) 
# snakemake all -n -s Snakefile --configfile config/snake_config_model.yml -q --rerun-incomplete 
#instead of rerun-incomplete, you could do forceall 

conda deactivate