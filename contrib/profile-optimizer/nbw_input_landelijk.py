import sys
import pandas as pd
# path to model generator
sys.path.append(r"..\..\benodigde gegevens\02_Modelgenerator\20200908\delft3dfmpy")
from delft3dfmpy import DFlowFMModel, HyDAMO, Rectangular, DFlowFMWriter
from delft3dfmpy import DFlowRRModel, DFlowRRWriter
from delft3dfmpy.datamodels.common import ExtendedGeoDataFrame
from delft3dfmpy.core import checks, geometry
from datetime import datetime
import os
from joblib import Parallel, delayed
import sys
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import shutil
import csv
import rasterio
import ogr
import gdal
from rasterstats import zonal_stats
import math
import subprocess
import time
import joblib
import joblib.parallel
import shutil
import fileinput
sys.path.append(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\modellen\basis_met_B_watergangen")

from fmmodel import make_fm_model, generate_fm

# inladen batch input
os.chdir(r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\code")
batch_input = pd.read_csv(r"../input/batch_input.csv")

#nu inladen in loop de resultaten van stedelijk model
path_in = r"D:\NBW_berekeningen\stedelijk"
files =[]
for year in range(1,56):
    files.append(path_in + "\\" + str(year))

#make folders
path_in_l = r"D:\NBW_berekeningen\landelijk"
for nr in range(1,56):
    print(nr)
    if not os.path.exists(path_in_l + "\\" + str(nr)):
        os.mkdir(path_in_l + "\\" + str(nr))


for nr in range(1,56):
    print(nr)
    i = nr - 1
    destination_dir = path_in_l + "\\" + str(nr)

    #lateral_discharge = pd.read_csv(path_in+"\\"+str(nr)+"\\results.csv")
    #lateral_discharge

    #lateral_discharge['time']= pd.to_datetime(lateral_discharge['time'])

    #lateral_discharge.index = lateral_discharge.time


    startdate = datetime.strptime(batch_input["start"][i], '%d/%m/%Y')
    refdate = startdate.strftime('%Y%m%d')
    enddate = datetime.strptime(batch_input["eind"][i], '%d/%m/%Y')
    Tstop = int((enddate - startdate).total_seconds())
    startdate_fm = startdate.strftime('%Y/%m/%d')
    enddate_fm = enddate.strftime('%Y/%m/%d')

    startdate_rtc = startdate.strftime('%Y-%m-%d')
    enddate_rtc = enddate.strftime('%Y-%m-%d')
    #dfmmodel.mdu_parameters['refdate'] = int(refdate)
    #dfmmodel.mdu_parameters['tstart'] = 0

    output_dir = path_in_l + "\\"+str(nr)


    #fm_writer = DFlowFMWriter(dfmmodel, output_dir=output_dir, name='import')

    #fm_writer.write_boundary_conditions()
    #fm_writer.write_laterals()
    #os.rename(output_dir+ "\\fm\\"+ 'boundaries.bc',output_dir+ "\\fm\\"+ 'import_lateral_sources.bc')

    #fm_writer.write_boundary_conditions()
    #os.rename(output_dir + "\\fm\\" + 'boundaries.bc', output_dir + "\\fm\\" + 'import_boundaryconditions1d.bc')
    # kopieren bestanden

    if batch_input["seizoen"][i] == "w":
        seizoen = "W"
        source_dir = r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\modellen\winter_opgeschoond"
    else:
        seizoen = "Z"
        source_dir = r"C:\Users\908800\Box\BH3371 NBW Oosterw\BH3371 NBW Oosterw WIP\04_opzet_model\NBW_doorrekening\modellen\zomer_opgeschoond"

    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

    # aanpassen bestanden

    filename = destination_dir + "\\dimr_config.xml"
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            replacement_text = "<time>0 3600 " + str(Tstop) + "</time>"
            print(line.replace("<time>0 300 86400</time>", replacement_text), end='')
    filename = destination_dir + "\\rr\\DELFT_3B.INI"

    filename = destination_dir + "\\rr\\DELFT_3B.INI"
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            replacement_text_s = "StartTime=" + startdate_fm + ";00:00:00"
            print(line.replace("StartTime=2020/01/01;00:00:00", replacement_text_s), end='')

    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            replacement_text_e = "EndTime=" + enddate_fm + ";00:00:00"
            print(line.replace("EndTime=2020/01/01;00:00:00", replacement_text_e), end='')

    filename = destination_dir + "\\fm\\import.mdu"

    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            replacement_text = "RefDate           = " + str(refdate)
            print(line.replace("RefDate           = 20170822", replacement_text), end='')

    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            replacement_text = "TStop             = " + str(Tstop)
            print(line.replace("TStop             = 86400", replacement_text), end='')

    filename = destination_dir + "\\rtc\\rtcRuntimeConfig.xml"

    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            replacement_text = startdate_rtc
            print(line.replace("2017-08-22", replacement_text), end='')

    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            replacement_text = enddate_rtc
            print(line.replace("2017-09-20", replacement_text), end='')

    filename = destination_dir + "\\fm\\Maxima.fou"
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            replacement_text = str(Tstop)
            print(line.replace("2588400", replacement_text), end='')

    filename = destination_dir + "\\fm\\import_boundaryconditions1d.bc"
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            replacement_text = "minutes since " + refdate[0:4]
            if seizoen == "W":
                print(line.replace("minutes since 2020", replacement_text), end='')
            else:
                print(line.replace("minutes since 2017", replacement_text), end='')


def run_bat(path):
    path_file = path + "\\"+ "run_bat.bat"
    subprocess.call([path_file],cwd=path)

files =[]
for year in range(1,56):
    files.append(path_in_l + "\\" + str(year))

Parallel(n_jobs=6)(
        delayed(run_bat)(
            file)
    for file in files
)

