#%% Import packages
from pathlib import Path 
import os, sys
import zipfile
import shutil
import sys

#%% extract params and file locations from snakemake
# SnakeMake arguments are automatically passed WITHOUT importing a lib. So this .py file is basicallly used as a template.
with open(snakemake.log[0], "w") as f:
    sys.stderr = sys.stdout = f

    fn_member = snakemake.params.fn_member
    member_name = snakemake.params.member_name
    dt = snakemake.params.name_dt
    var = snakemake.params.name_var

    fn_out = snakemake.output
    fn_in = snakemake.input.f_zip

    to_extract = os.path.join("/full_ds", fn_member,var, f"{var}.KNMI-{year}.KEXT12.kR2v3-v578-fECEARTH3-ds23-{member_name}+hist.DD.nc")

    print('Member: {}'.format(member_name))
    print('DT: {}'.format(dt))
    print('VAR: {}'.format(var))
    print('fn_out: {}'.format(fn_out))
    print('fn_in: {}'.format(fn_in))
    print('to_extract: {}'.format(to_extract))

    #%%
    #fn_in = r'p:\11208719-interreg\data\racmo\members_bias_corrected\a_raw\daily\daily_data_15022023.zip'
    #fn_out = r'p:\11208719-interreg\data\racmo\members_bias_corrected\a_raw\daily'

    # #%%
    # #We extract the name of folder to remove before member string
    # dirs = []
    # with zipfile.ZipFile(str(fn_in), "r") as zip_ref:
    #     # Loop over each file in the .zip archive and extract it
    #     for member in zip_ref.infolist():
    #         # Check if the current member is a file (not a directory)
    #         if member.is_dir():
    #             dirs.append(member.filename)

    # #We remove one level if it is not relevant
    # dir_name = dirs[0]
    # if "Member" or "member" in dir_name:
    #     dir_name = dirs[0]
    # else:
    #     dir_name = ''

    with zipfile.ZipFile(str(fn_in), "r") as zip_ref:
        print("Extracting {}".format(member.filename))
        zip_ref.extract(to_extract, path=str(fn_out))    

    print("Done!")             

    # # Open the .zip file using the ZipFile object
    # with zipfile.ZipFile(str(fn_in), "r") as zip_ref:
    #     # Loop over each file in the .zip archive and extract it
    #     for member in zip_ref.infolist():
    #         # Check if the current member is a file (not a directory)
    #         if not member.is_dir():
    #             # print("In the loop")
    #             #Check whether this is a variable we want to extract:
    #             # print("Parent is ", os.path.basename(Path(member.orig_filename).parents[1]))
    #             # print("Member is ", member_name)
    #             if os.path.basename(Path(member.orig_filename).parents[1]) == member_name: 
    #                 # print("Parent is ", os.path.basename(Path(member.orig_filename).parents[0]))
    #                 # print("Var is ", var)                    
    #                 if os.path.basename(Path(member.orig_filename).parents[0]) == var: 
    #                     # Open the member and read its contents
    #                     print("Extracting {}".format(member.filename))
    #                     zip_ref.extract(member, path=str(fn_out))                

    #                     # if dir_name != '': #We move the file if there is one level higher and create the directory if needed
    #                     #     if not os.path.exists(os.path.dirname(os.path.join(fn_out,member.orig_filename.replace(f"{dir_name}", '')))):
    #                     #         os.makedirs(os.path.dirname(os.path.join(fn_out,member.orig_filename.replace(f"{dir_name}", ''))))
    #                     #     shutil.move(os.path.abspath(os.path.join(fn_out,member.filename)), os.path.abspath(os.path.join(fn_out,member.orig_filename.replace(f"{dir_name}", ''))))            
    #                     #     print("Move done")

    # print("Unzipping finished for member {}, var {}".format(member_name, var))