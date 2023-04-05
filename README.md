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
        ├── 0-setup             <- Install necessary software, dependencies, pull other git projects, etc.
        ├── 1-prepare           <- Scripts and programs to process data, from 1-external to 2-interim.
        ├── 2-build             <- Scripts to create model specific inputm from 2-interim to 3-input. 
        ├── 3-model             <- Scripts to run model and convert or compress model results, from 3-input to 4-output.
        ├── 4-analyze           <- Scripts to post-process model results, from 4-output to 5-visualization.
        └── 5-visualize         <- Scripts for visualisation of your results, from 5-visualization to ./report/figures.