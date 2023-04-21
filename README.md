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
    ├── snakefile_cluster.sh    <- bash script to run on the cluster
    ├── config                  <- Configuration files, (e.g.: snakemake .yml config files)   
    ├── envs                    <- conda compatible .yml file for installation of HydroMT + Snake python virtual environment and workflows
    ├── docs                    <- Documentation, e.g., doxygen or scientific papers (not tracked by git)
    ├── notebooks               <- Jupyter notebooks
    └── src                     <- Source code for this project
        ├── preprocess 
        ├── model_building 
        ├── postprocess 

Information
--------------------
To find more information on how to run the snakemake pipeline on the cluster, you can have a look at ./notebooks/run_cluster.ipynb for tips