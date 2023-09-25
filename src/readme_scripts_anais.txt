- all the statistics are based on the wflow runs:
-- p:\11208719-interreg\wflow\p_geulrur\members_bias_corrected_revised_daily\ # for the daily bias corrected RACMO data
-- p:\11208719-interreg\wflow\p_geulrur\members_bias_corrected_revised_hourly\ # for the hourly bias corrected RACMO data

- all the scripts used to make the figures in the report are in src\postprocess\
- Figures have been saved under 
-- p:\11208719-interreg\Figures\p_geulrur\members_bias_corrected_revised_daily\ #For the daily data and general figures
-- p:\11208719-interreg\Figures\p_geulrur\members_bias_corrected_revised_hourly\ #For the hourly data 
-- in those folders there is a subfolder with \data to create some of the figures  


You can find the info to generate:
-Figure 1.1 --> n:\Projects\11208500\11208719\C. Report - advise\Reporting\Figures.pptx
-Figure 2.1 --> ..\src\postprocess\comparison_observations.py
-Figure 5.2/5.6/5.9/5.10 --> ..\src\postprocess\eva_analysis_figure.py that uses ..\src\postprocess\eva_analysis_functions.py and ..\src\postprocess\eva_analysis.py
-Figure 5.3/5.6 --> ..\src\postprocess\spatial_return_periods_figure.py that uses ..\src\postprocess\spatial_return_periods.py
-Figure 5.7/5.12 --> ..\src\postprocess\eva_analysis.py that uses ..\src\postprocess\eva_analysis_functions.py

-Table 5-1 and Table 5-2 --> reading the dictionaries from which we export the csv:
- src/postprocess/export_gev_table_results.py
-FOR THE DAILY
--p:\11208719-interreg\Figures\p_geulrur\members_bias_corrected_revised_daily\data\daily_results_stations.pickle #Dictionary with resutls per stations
--p:\11208719-interreg\Figures\p_geulrur\members_bias_corrected_revised_daily\data\daily_GEV_..._AMs.csv #We extract the data from the dictionary to have the tables
-FOR THE HOURLY
--p:\11208719-interreg\Figures\p_geulrur\members_bias_corrected_revised_hourly\data\hourly_results_stations.pickle #Dictionary with resutls per stations
--p:\11208719-interreg\Figures\p_geulrur\members_bias_corrected_revised_hourly\data\hourly_GEV_..._AMs.csv #We extract the data from the dictionary to have the tables

-Figure 5-11 --> ..\src\postprocess\eva_diff_dt_stations.py
-Figure 5.4 --> ..\src\postprocess\scatter_shape_catchment.py

Plots of the July 2021 events are done in:
- ..\src\postprocess\plot_july_2021.py

