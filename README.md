interreg-meuse
==============================

extreme streamflow statistics meuse

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