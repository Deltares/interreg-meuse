- all the model building steps are in interreg-meuse\src\model_building\ the latest one was o_rwsinfo, but Alessia has updated now to p_geulrur.
- calibration scripts are here: interreg-meuse\src\model_building\calibration_linux\

- Scripts used to process waterschap limburg and hydroportail data:
	interreg-meuse\src\preprocess\process_hydroportail_data.py
	interreg-meuse\src\preprocess\process_waterschap_limburg_data.py

Scripts used to make figures in the report: (all in interreg-meuse\src\postprocess\)

- Script to make fig 4.2 tot 4.4:
	plot_compare_runs_daily.py
	
	activate: 
	# "kinematic": {"case":"d_manualcalib", 
    #             "folder": "run_manualcalib_daily_eobs24_kinematic"},
    
    # "loc.iner": {"case": "d_manualcalib",
    #          "folder": "run_manualcalib_daily_eobs24_1d"},
    
    # "loc.iner.flpl1d": {"case":"d_manualcalib", 
    #             "folder": "run_manualcalib_daily_eobs24"},
    
    # "loc.iner1d2d": {"case": "d_manualcalib",
    #          "folder": "run_manualcalib_daily_eobs24_1d2d"},
	
	
	deactivate
	    "before": {"case": "j_waterschaplimburg",
             "folder": "run_waterschaplimburg_eobs25"},

    # #after cal 02
    # "after": {"case": "l_snakecal02",
    #          "folder": "run_snakecal02_eobs25"},

    # #after cal 03
    "after": {"case": "m_snakecal03",
             "folder": "run_snakecal03_eobs25"},
			 
	
	activate 
	caserun = "routing2"   
	
	comment out: (line 247)
	dsq = ds['Q'].sel(index = station_id, runs = runs_sel + ["Obs."]).sel(time = slice('2006-01-01', '2017-12-31')).to_dataset().dropna(dim='time')
	
	
- Figure 4.5 -- calibration figures for all other catchments are found here:
	p:\11208719-interreg\wflow\j_waterschaplimburg\runs_calibration_linux_01\Results\Plots\
	

	
- Figure 4.6  - cal maps:
	interreg-meuse\src\model_building\calibration_linux\scripts\plot_param_maps.py
	
	
	
- Figure 4.7 and further for the daily model results:
	plot_compare_runs_daily_per_source.py
	
- Figure 4.23 and further for the hourly model:
	plot_compare_runs_hourly_per_source.py
	
	
- Figure 4.37 4.38 4.39 4.40
	plot_compare_runs_daily.py
	plot_compare_runs_hourly.py
	


- July 2021 analysis was done in: 
	p:\11208719-interreg\wflow\m_snakecal03\run_snakecal03_july_2021\
	p:\11208719-interreg\wflow\m_snakecal03\run_snakecal03_july_2021_1d\
	p:\11208719-interreg\wflow\m_snakecal03\run_snakecal03_july_2021_1d2d\
	p:\11208719-interreg\wflow\m_snakecal03\run_snakecal03_july_2021_kinematic\

	plots in: 
	p:\11208719-interreg\wflow\m_snakecal03\plots_july_2021_routing_model_m_snakecal03\

	script in: interreg-meuse\src\postprocess\juli2021_poster.py
	(not all lines though....)