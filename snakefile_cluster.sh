#!/bin/bash
#$ -q normal-e3-c7
#$ -cwd
#$ -m bea
#$ -pe distrib 1
#$ -V
#$ -N snake_wflow3
#$ -j yes


# Initiating snakemake and running workflow in cluster mode
source /u/bouaziz/anaconda/bin/activate snakeymakey

conda config --set channel_priority strict

ROOT="/u/bouaziz/interreg-meuse"
cd "${ROOT}"

snakemake --unlock -s snakefile --configfile config/members_config.yml 
# snakemake -s snakefile --configfile config/members_config.yml --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --rerun-triggers mtime
snakemake -s snakefile --configfile config/members_config.yml --batch all=1/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=2/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=3/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=4/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=5/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=6/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=7/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=8/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=9/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=10/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=11/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=12/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=13/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=14/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=15/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 
snakemake -s snakefile --configfile config/members_config.yml --batch all=16/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3102 xr_merge=50 

conda deactivate