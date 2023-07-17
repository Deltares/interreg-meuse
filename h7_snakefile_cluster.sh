#!/bin/bash
#SBATCH --job-name=daily_interreg          # Job name
#SBATCH --output=daily_run_%j.log     # Standard output and error log
#SBATCH --time=0-24:00:00           # Job duration (hh:mm:ss)
#SBATCH --partition 1vcpu
#SBATCH --ntasks=1                  # Number of tasks (analyses) to run
#SBATCH --mail-user=anais.couasnon@deltares.nl
#SBATCH --mail-type=ALL
#SBATCH --get-user-env





# Initiating snakemake and running workflow in cluster mode
source /u/couasnon/miniconda3/bin/activate hydromt-wflow
conda config --set channel_priority strict

#Going to the folder where scripts are
ROOT="/u/couasnon/git_repos/interreg-meuse"
cd "${ROOT}"

#Unlocking the directory for snakemake
snakemake --unlock -s snakefile --configfile config/members_config.yml 

#All the cluster configuration (both general and rule specific) is in ~/.config/snakemake/simple 
snakemake -s snakefile --configfile config/members_config.yml --profile interreg_daily/ --wait-for-files --directory $PWD  --rerun-triggers mtime #--group-components preprocess=3120 xr_merge=50  #--retries 2 --allowed-rules run_wflow
#snakemake -s snakefile --configfile config/members_config.yml --cluster "sbatch --time=0-00:20:00 --partition 4vcpu --cpus-per-task=4" --jobs 20 --latency-wait 60 --wait-for-files --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --allowed-rules run_wflow #--retries 2 --profile simple/

conda deactivate












#Previous setup using batch but not so useful here
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=1/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=2/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=3/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=4/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=5/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=6/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=7/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=8/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=9/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=10/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=11/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=12/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=13/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=14/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=15/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
#snakemake -s snakefile_nocdo --configfile config/members_config.yml --batch all=16/16 --cluster "$SNAKE_SUBMIT_JOB" --latency-wait 60 --wait-for-files --jobs 20 --use-conda --directory $PWD --keep-going --group-components preprocess=3120 xr_merge=10 --retries 2 --rerun-incomplete
