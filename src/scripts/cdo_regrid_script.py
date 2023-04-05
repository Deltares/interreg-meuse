import os, glob
import glob
from cdo import *
import sys

#Initialize cdo
cdo = Cdo()

# #Define source and destination folder
#We parse the snakemake params
files = snakemake.input
Folder_src = snakemake.params.f_src
Folder_dst = snakemake.params.f_dst
grid = snakemake.params.grid_fn
variable = snakemake.params.var_name
dt = snakemake.params.dt_step
outfile_path = str(snakemake.output)

print("------- Checking what we got ------")
print("Outfile path", outfile_path)
print("file", files)
print("Folder_src", Folder_src)
print("folder_dst", Folder_dst)
print("grid", grid)
print("variable", variable)
print("dt", dt)
print("Performing timestep {}".format(dt))

for file in files:
    print("Doing file: ", file)
    if variable == "t2m":
        cdo.remapnn(grid, input='-selvar,t2m {}'.format(str(file)), output=outfile_path, options = "-f nc") 

    if variable == "precip":
        cdo.remapnn(grid, input='-setrtoc,-999.0,0,0  -selvar,precip {}'.format(str(file)), output=outfile_path, options = "-f nc") 
    #            cdo.setrtoc(input='-999.0,0,0 -remapnn,{} -selvar,precip {}'.format(grid, file), output=outfile_path, options = "-f nc") 

    if variable == "pet":
        cdo.remapnn(grid, input='-setrtoc,-999.0,0,0 {}'.format(str(file)), output=outfile_path, options = "-f nc") 

    print("File done")  

print("Done for timestep {} and variable {}".format(dt, variable))
