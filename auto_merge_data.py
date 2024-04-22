# This script was created to automatically post-process P.L.U.M.E. Van data after multiple days of sampling
# Function: merges "Sensor transcript" data and "GPS data"
# Authors: Rachel Habermehl, Chris Kelly, Davi de Ferreyro Monticelli, iREACH group (University of British Columbia)
# Date: 2023-06-19
# Version: 1.0.0

# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "Pre-processing_PLUME_Data.py" PRIOR

import pandas as pd
import csv
from datetime import datetime
from dateutil import tz
from dateutil.relativedelta import relativedelta
import re
import shutil
import numpy as np
import os
import sys
import io
import time
# Get the current user's username
username = os.getlogin()
########################################################################################################################
# Creating a custom class to save console printed messages in a single file:
# Essentially, the created file "XXX_console_output.txt" will have all printed messages from this script

class Tee(io.TextIOWrapper):
    def __init__(self, file, *args, **kwargs):
        self.file = file
        super().__init__(file, *args, **kwargs)

    def write(self, text):
        self.file.write(text)
        sys.__stdout__.write(text)  # Print to the console

# Specify the file path where you want to save the output
file_path_txt = f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Merge_AQ_and_GPS_console_output.txt'

# Open the file in write mode and create the Tee object
output_file_txt = open(file_path_txt, 'w')
tee = Tee(output_file_txt)

# Redirect the standard output to the Tee object
sys.stdout = tee

########################################################################################################################
# Simply checking how long it takes to execute the script

start_time = time.time()
start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print("Start time of run: ", start_time_formatted)

########################################################################################################################
# MAIN

# Function: Create a dataframe variable for each sensor transcript in folder "Files"
def read_Dash_files(date_to_run):
    # Read Sensor_Transcript_XXXX_XX_XX csv files
    date = date_to_run
    file_path = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\"
    files_updated = f"Sensor_Transcript_{date}_UPDATED.csv"
    sensor_data = pd.read_csv(file_path+files_updated)

    return sensor_data

# File renaming:
# Files needed: GPS (.gpx) files
# Function: rename all files in the folder to match the expected format ("Something_YYYY_MM_DD") to run the script

# Specify the path to the folder containing the files
folder_path = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\inputs"

# Get a list of file names in the folder
file_names = os.listdir(folder_path)
old_files = file_names
print("Old file names are:", file_names)

# Iterate over the file names and rename each file
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    destine_path = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\renamed"

    # Rename "GPX" files
    if file_name.endswith("GL770.gpx"):
        date_str = file_name.split("_")[0].replace(".gpx", "")
        parts = date_str.split("-")
        new_file_name = f"GPS_20{parts[2]}_{parts[1]}_{parts[0]}.gpx"
        new_file_path = os.path.join(destine_path, new_file_name)
        shutil.copy(file_path, new_file_path)

# Specify the path to the folder containing the renamed GPS files
folder_path = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\renamed"

# Get a list of file names in the folder
file_names = os.listdir(folder_path)
print("New file names are:", file_names)

# Get Dash file names and dates
source_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files"
file_names = os.listdir(source_folder)

# Sensor transcript data (file names):
# These files were written by the "Pre-processing_PLUME_Data.py" script
Sensor_names = [file_name for file_name in file_names if re.match(r"Sensor_Transcript_\d{4}_\d{2}_\d{2}_UPDATED.csv", file_name)]
Sensor_names = [os.path.splitext(file_name)[0] for file_name in Sensor_names]  # Get rid of the .csv
# Checking:
print("Dash files are:", Sensor_names)

# Get GPX file names and dates
source_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\renamed"
file_names = os.listdir(source_folder)

# GPS data (file names):
GPS_names = [file_name for file_name in file_names if re.match(r"GPS_\d{4}_\d{2}_\d{2}.gpx", file_name)]
GPS_names = [os.path.splitext(file_name)[0] for file_name in GPS_names]  # Get rid of the .gpx
# Checking:
print("GPS files are:", GPS_names)

# Get the unique dates from the Sensor_transcript
# Remove the "_UPDATED" portion from Sensor names string:
Sensor_names_clean = [s.replace("_UPDATED", "") for s in Sensor_names]
# Dates:
Sensor_dates = list(set([re.sub(r"Sensor_Transcript_", "", var_name) for var_name in Sensor_names_clean]))
GPS_dates = list(set([re.sub(r"GPS_", "", var_name) for var_name in GPS_names]))

# Find the dates that are common to both Sensor (Dash) and AQ monitors dataframes
common_GPS_dates = list(set(Sensor_dates) & set(GPS_dates))
print("Sensor + GPS common dates are:", common_GPS_dates)
print("")

# Informing the lag time (IN SECONDS) between sampling (inlet) and instrument reading (measurement)
# [[[ INFORMED BY THE USER ]]]
lags = {
  'no2': 40,
  'wcpc': 5,
  'o3': 40,
  'co': 40,
  'co2': 40,
  'no': 40
  }

max_lag = 0
for pollutant in lags:
  if lags[pollutant] > max_lag:
    max_lag = lags[pollutant]

# Read Dash files and create proper dataframes:
for date in Sensor_dates:
    globals()["Sensor_Transcript" + date] = read_Dash_files(date)
    # Rename columns
    globals()["Sensor_Transcript" + date].columns = ['date', 'NO2 (ppb)', 'UFP (#/cm^3)', 'O3 (ppb)', 'CO (ppm)', 'CO2 (ppm)', 'NO (ppb)', 'WS (m/s)', 'WD (degrees)', 'WV (m/s)']
    # First row is the column names
    globals()["Sensor_Transcript" + date] = globals()["Sensor_Transcript" + date].drop(globals()["Sensor_Transcript" + date].index[0]).reset_index(drop=True)
    # Adjust time format to datetime
    globals()["Sensor_Transcript" + date]['date'] = pd.to_datetime(globals()["Sensor_Transcript" + date]['date'], format='%Y-%m-%d %H:%M:%S')

    # Adjust data format to float (except column date)
    globals()["Sensor_Transcript" + date] = globals()["Sensor_Transcript" + date].astype({col: float for col in globals()["Sensor_Transcript" + date].columns if col != 'date'})
    # Getting the Dash file start and end times:
    globals()["Dash_start_time" + date] = globals()["Sensor_Transcript" + date]['date'].iloc[0]
    globals()["Dash_end_time" + date] = globals()["Sensor_Transcript" + date]['date'].iloc[-1]
    globals()["Dash_end_time" + date] = globals()["Dash_end_time" + date] - relativedelta(seconds=max_lag)
    # Checking
    print("Dashboard output for date ", date, " starts at: ", globals()["Dash_start_time" + date], "and ends at: ", globals()["Dash_end_time" + date])
    print(globals()["Sensor_Transcript" + date])

############################################# Code for GL770 GPS data ##################################################
source_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\renamed\\"

for date in GPS_dates:
    # Converting from GPX to csv
    data = open(source_folder+(f"GPS_{date}.gpx")).read()
    lat = np.array(re.findall(r'lat="([^"]+)', data), dtype = float)
    lon = np.array(re.findall(r'lon="([^"]+)', data), dtype = float)
    time = re.findall(r'<time>([^\<]+)', data)
    combined = np.array(list(zip(lat, lon, time)))
    df = pd.DataFrame(combined)
    df.columns = ['latitude', 'longitude', 'time']
    df.to_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\aux_files\\GPS_{date}.csv')
    # Changing from UTC time to Pacific time
    with open(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\aux_files\\GPS_{date}.csv', 'r') as csv_file:
       csv_reader = csv.DictReader(csv_file)
       times = []
       for col in csv_reader:
           times.append(col['time'])

    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    for i in range(len(times)):
        t = str(times[i]).split(".")[0]
        t = str(t).split("Z")[0]
        utc_dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
        utc_dt = utc_dt.replace(tzinfo=from_zone)
        pst_dt = utc_dt.astimezone(to_zone)
        times[i] = pst_dt.strftime("%Y-%m-%d %H:%M:%S")

    times.sort()

    # Writing the new times (plus lat and long) back to a csv
    all_data = np.array(list(zip(lat, lon, times)))
    all_data_df = pd.DataFrame(all_data)
    all_data_df.columns = ['latitude', 'longitude', 'time']
    all_data_df.to_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\aux_files\\GPS_{date}_final.csv')

    # Convert to datetime so that it can merge with dashboard data
    globals()["GPS_" + date] = pd.read_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\aux_files\\GPS_{date}_final.csv')
    globals()["GPS_" + date]['time'] = pd.to_datetime(globals()["GPS_" + date]['time'], errors='coerce')
    globals()["GPS_" + date].rename(columns={'time': 'date'}, inplace=True)

############################################# Code for Merging GPS + Dash data #########################################

for date in common_GPS_dates:
    # Import dashboard data
    dash_data = globals()["Sensor_Transcript" + date]
    gps_data = globals()["GPS_" + date]
    # Merge dashboard and GL770 GPS data
    merged_data = pd.merge_asof(gps_data, dash_data, on='date', direction='nearest')
    # Fixing time-lag
    # For mobile air quality monitoring, one needs to account the lag between air entering the inlet and concentration read by the instruments
    # Below one can insert as variables the 'lag time' for each instrument
    # What follows can be cut or added for every new pollutant
    # Finds column with name '...' and shift its values down by a lag. If desired to shift up, include minus sign
    merged_data['NO2 (ppb)'] = merged_data['NO2 (ppb)'].shift((-1)*lags['no2'])
    merged_data['O3 (ppb)'] = merged_data['O3 (ppb)'].shift((-1)*lags['o3'])
    merged_data['CO (ppm)'] = merged_data['CO (ppm)'].shift((-1)*lags['co'])
    merged_data['NO (ppb)'] = merged_data['NO (ppb)'].shift((-1)*lags['no'])
    merged_data['CO2 (ppm)'] = merged_data['CO2 (ppm)'].shift((-1)*lags['co2'])
    merged_data['UFP (#/cm^3)'] = merged_data['UFP (#/cm^3)'].shift((-1)*lags['wcpc'])
    # Cut-off first and last rows of obsolete data
    # Because GPS should be started before running the Dashboard, it will be logging first
    # When finding the nearest time to merge the Air Quality Data, the program will keep
    # repeating the first data row from the Dashboard log until the Dash time hits.
    # Below one can filter the specific time to account both (and only) Dash and GPS synced data
    merged_data = merged_data[~(merged_data['date'] < globals()["Dash_start_time" + date])]
    merged_data = merged_data[~(merged_data['date'] > globals()["Dash_end_time" + date])]
    # Drop first column (redundant, just leftover indexes)
    merged_data = merged_data.iloc[:, 1:]
    # Export final merged dataframe as a .csv
    print("")
    print(f"For {date} new AQ+GPS merged file is:")
    print("")
    print(merged_data)
    merged_data.to_csv(f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\MERGED_AQ_GPS_{date}.csv', index=False)

# Ending remarks:
print("")
end_time = time.time()
end_time_formatted = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
print("End time is: ", end_time_formatted)
print("Elapsed time is %.2f seconds." % round(end_time - start_time, 2))

# Restore the standard output
sys.stdout = sys.__stdout__

# Close the output file
output_file_txt.close()


'''
    ####################################################
    # Debugging:
    # Finding strings in dataframe (should be time and float only)
    string_columns = globals()["Sensor_Transcript" + date].select_dtypes(include='object').columns
    matching_rows = globals()["Sensor_Transcript" + date][globals()["Sensor_Transcript" + date][string_columns].notnull().any(axis=1)]

    # Print the matching rows
    print("Matching Rows:")
    print(matching_rows)

    # Print the row with the string value found
    string_row = globals()["Sensor_Transcript" + date][string_columns].stack().loc[lambda x: x == x]
    print("Row with String Value:")
    print(string_row)
    ####################################################
'''