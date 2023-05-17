call activate hydromt-wflow

rem snakemake -s Snakefile --configfile config/snake_config_model_windows.yml --dag | dot -Tpng > dag_all.png

snakemake --unlock -s Snakefile --configfile config/snake_config_model_windows.yml
snakemake all -c 1 -s Snakefile --configfile config/snake_config_model_windows.yml

pause
