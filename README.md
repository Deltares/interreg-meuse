Interreg-meuse
==============================

This repository contains the code produced by Deltares for the EMFloodResilience project within the Interreg V-A Euroregio. The code was written on behalf of Rijkswaterstaat under the EMfloodResilience project. The results from the project and report are available at: https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@268763/evaluation-discharge-extremes-the-meuse/  

Acknowledgments
--------------------
The EMfloodResilience project is being carried out within the context of Interreg V-A Euregio MeuseRhine and is 90% funded from the European Regional Development Fund.

Project Organization
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── snakefile               <- Snakemake workflow file(s)
    ├── snakefile_cluster.sh    <- bash script to run on the cluster on h6
    ├── h7_snakefile_cluster.sh <- bash script to run on the cluster on h7
    ├── config                  <- Configuration files, (e.g.: snakemake .yml config files)   
    ├── envs                    <- conda compatible .yml file for installation of HydroMT + Snake python virtual environment and workflows
    ├── docs                    <- Documentation, e.g., doxygen or scientific papers (not tracked by git)
    ├── notebooks               <- Jupyter notebooks
    └── home                    <- Snakemake .config folder. Content should be placed in ~/.config/snakemake
        ├── interreg_daily 
        ├── interreg_hourly 
    └── src                     <- Source code for this project
        ├── preprocess 
        ├── model_building 
        ├── postprocess 

Information
--------------------
To find more information on how to run the snakemake pipeline on the cluster (h6 or h7), you can have a look at ./notebooks/run_cluster_h6.ipynb 
and ./notebooks/run_cluster_h7.ipynb for tips
