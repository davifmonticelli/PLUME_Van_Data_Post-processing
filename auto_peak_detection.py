# This script was created to automatically post-process P.L.U.M.E. Van data after multiple days of sampling
# Function: Uses the merged (Sensor transcript + GPS) to markdown peaks in the timeseries
# Authors: Chris Kelly and Davi de Ferreyro Monticelli, iREACH group (University of British Columbia)
# Date: 2023-06-22
# Version: 1.0.0

# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "Pre-processing_PLUME_Data.py" PRIOR
# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "auto_merge_data.py" PRIOR
# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "auto_baseline.py" PRIOR IF YOU WISH TO PLOT ISOLATED PEAKS

import math
import csv
import pandas as pd
from collections import deque
import statistics
import numpy as np
import sys
import io
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import find_peaks, peak_prominences
import time
from datetime import datetime
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
file_path_txt = f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Peaks\\Peak_console_output.txt'

# Open the file in write mode and create the Tee object
output_file_txt = open(file_path_txt, 'w')
tee = Tee(output_file_txt)

# Redirect the standard output to the Tee object
sys.stdout = tee

########################################################################################################################

start_time = time.time()
start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print("Start time of run: ", start_time_formatted)

########################################################################################################################
# Settings declaration:
########################################################################################################################
# Please refer to the Dashboard paper and supporting information found at: https://doi.org/10.1016/j.envsoft.2022.105600
# to understand how and when to adjust these settings

# A1_coeff settings:
no2_coeff = 15
ufp_coeff = 15
o3_coeff = 15
co_coeff = 15
co2_coeff = 15
no_coeff = 15

# A1_percentile settings:
no2_per = 50
ufp_per = 50
o3_per = 50
co_per = 50
co2_per = 50
no_per = 50

# A1_thresh_bump_percentile settings:
no2_tbump = 1
ufp_tbump = 1
o3_tbump = 1
co_tbump = 1
co2_tbump = 1
no_tbump = 1

# Get Merged AQ + GPS file names and dates
source_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\"
file_names = os.listdir(source_folder)

# Merged AQ + GPS data (file names):
Merged_names = [file_name for file_name in file_names if re.match(r"MERGED_AQ_GPS_\d{4}_\d{2}_\d{2}.csv", file_name)]
Merged_names = [os.path.splitext(file_name)[0] for file_name in Merged_names]  # Get rid of the .csv
print("Files processed are: ", Merged_names)
# Dates:
Merged_dates = list(set([re.sub(r"MERGED_AQ_GPS_", "", var_name) for var_name in Merged_names]))
print("Dates processed are: ", Merged_dates)

# A1_misc settings
startup_bypass = 30
folder_directory = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\"
input_filenames = [file_name for file_name in file_names if re.match(r"MERGED_AQ_GPS_\d{4}_\d{2}_\d{2}.csv", file_name)]
output_filenames = ['Peaks_in_' + item for item in input_filenames]
chunk_size = 3000
print("Input file names are: ", input_filenames)
print("Output file names will start with: ", output_filenames)

# A1_post_processing_thresh_dump settings:
no2_post_tbumb = True
ufp_post_tbumb = True
o3_post_tbumb = True
co_post_tbumb = True
co2_post_tbumb = True
no_post_tbumb = True

# A1_bulk_processing settings:
enable_bulk_processing = True
no2_coeffs = [5, 10, 25]
no2_percentiles = [50]
no2_thresh_bump_percentiles = [1]
ufp_coeffs = [5, 10, 25]
ufp_percentiles = [50]
ufp_thresh_bump_percentiles = [1]
o3_coeffs = [5, 10, 25]
o3_percentiles = [50]
o3_thresh_bump_percentiles = [1]
co_coeffs = [5, 10, 25]
co_percentiles = [50]
co_thresh_bump_percentiles = [1]
co2_coeffs = [5, 10, 25]
co2_percentiles = [50]
co2_thresh_bump_percentiles = [1]
no_coeffs = [5, 10, 25]
no_percentiles = [50]
no_thresh_bump_percentiles = [1]

######################################################################################################################################
# ALTERNATIVE METHOD (NEW FEATURE)
# If True, it will use the find_peak() function from scipy.signal package
# Please see more information about this method at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
# It also automatically plot the results
# If this method is used the MAIN method is overruled
# Advantages: it is faster and provide good results
# Disadvantages: work with the following fixed configuration (* = must be known and informed by the user):
#  - Minimum peak height: 110% of the minimum pollutant concentration value in timeseries
#  * Peak distance: the update time of the instrument (in seconds)
#  - Prominence: 5% of the highest prominence for the peaks found first using only "height" and "distance"
alternative_peaks = True
#######################################################################################################################################

# Plotting results?
plotting_results = True
# Plotting isolated peaks (from baseline subtraction)?
plotting_from_baseline = True

########################################################################################################################
# Declaring functions (helpers) for the script
########################################################################################################################

# Fills in settings for smaller bulk processing list
def fill_smaller_bulk(smaller,target_length):
    while len(smaller) < target_length:
        smaller.append(smaller[-1])
    return smaller

########################################################################################################################
# GLOBAL SETTINGS
########################################################################################################################

# Grabbing directory and filename, fixing formatting, checking if input file exists at that directory
directory = folder_directory

bulk_processing = enable_bulk_processing

A1_thresh_dump = {
        "no2": no2_post_tbumb,
        "ufp": ufp_post_tbumb,
        "o3": o3_post_tbumb,
        "co": co_post_tbumb,
        "co2": co2_post_tbumb,
        "no": no_post_tbumb
     }

# Grabbing pollutant specific settings, either for bulk or normal
if not bulk_processing:
    A1_coeff = {
        "no2": no2_coeff,
        "ufp": ufp_coeff,
        "o3": o3_coeff,
        "co": co_coeff,
        "co2": co2_coeff,
        "no": no_coeff
    }
    A1_percentile = {
        "no2": no2_per,
        "ufp": ufp_per,
        "o3": o3_per,
        "co": co_per,
        "co2": co2_per,
        "no": no_per
    }
    A1_thresh_bump_percentile = {
        "no2": no2_tbump,
        "ufp": ufp_tbump,
        "o3": o3_tbump,
        "co": co_tbump,
        "co2": co2_tbump,
        "no": no_tbump
    }
else:
    max_entries = 0
    all_A1_coeffs = {
        "no2": no2_coeffs,
        "ufp": ufp_coeffs,
        "o3": o3_coeffs,
        "co": co_coeffs,
        "co2": co2_coeffs,
        "no": no_coeffs
    }
    coeff_crits = {
        "no2": True,
        "ufp": True,
        "o3": True,
        "co": True,
        "co2": True,
        "no": True
    }
    for i in coeff_crits:
        if all_A1_coeffs[i][-1] == '*':
            coeff_crits[i] = True
            del all_A1_coeffs[i][-1]
    for i in all_A1_coeffs:
        if (len(all_A1_coeffs[i]) > max_entries):
            max_entries = len(all_A1_coeffs[i])

    all_A1_percentiles = {
        "no2": no2_percentiles,
        "ufp": ufp_percentiles,
        "o3": o3_percentiles,
        "co": co_percentiles,
        "co2": co2_percentiles,
        "no": no_percentiles
    }
    percentile_crits = {
        "no2": True,
        "ufp": True,
        "o3": True,
        "co": True,
        "co2": True,
        "no": True
    }
    for i in percentile_crits:
        if all_A1_percentiles[i][-1] == '*':
            percentile_crits[i] = True
            del all_A1_percentiles[i][-1]
    for i in all_A1_percentiles:
        if (len(all_A1_percentiles[i]) > max_entries):
            max_entries = len(all_A1_percentiles[i])

    all_A1_thresh_bump_percentiles = {
        "no2": no2_thresh_bump_percentiles,
        "ufp": ufp_thresh_bump_percentiles,
        "o3": o3_thresh_bump_percentiles,
        "co": co_thresh_bump_percentiles,
        "co2": co2_thresh_bump_percentiles,
        "no": no_thresh_bump_percentiles
    }
    thresh_bump_crits = {
        "no2": True,
        "ufp": True,
        "o3": True,
        "co": True,
        "co2": True,
        "no": True
    }
    for i in thresh_bump_crits:
        if all_A1_thresh_bump_percentiles[i][-1] == '*':
            thresh_bump_crits[i] = True
            del all_A1_thresh_bump_percentiles[i][-1]
    for i in all_A1_thresh_bump_percentiles:
        if (len(all_A1_thresh_bump_percentiles[i]) > max_entries):
            max_entries = len(all_A1_thresh_bump_percentiles[i])

    # Filling in smaller bulk settings to meet max entries length
    for i in all_A1_coeffs:
        all_A1_coeffs[i] = fill_smaller_bulk(all_A1_coeffs[i],max_entries)

    for i in all_A1_percentiles:
        all_A1_percentiles[i] = fill_smaller_bulk(all_A1_percentiles[i],max_entries)

    for i in all_A1_thresh_bump_percentiles:
        all_A1_thresh_bump_percentiles[i] = fill_smaller_bulk(all_A1_thresh_bump_percentiles[i],max_entries)

    # Handling output names based on crit settings or if no crit settings are selected
    contains_crits = False
    for i in coeff_crits:
        if coeff_crits[i]:
            contains_crits = True
            break
        if percentile_crits[i]:
            contains_crits = True
            break
        if thresh_bump_crits[i]:
            contains_crits = True
            break

    filename_no_extension = 'Bulk_settings'
    bulk_settings = []

    if contains_crits:
        for i in range(0,max_entries):
            working_output = filename_no_extension

            for x in coeff_crits:
                if coeff_crits[x]:
                    working_output+=', '+str(x)+'_coeff='+str(all_A1_coeffs[x][i])

            for x in percentile_crits:
                if percentile_crits[x]:
                    working_output+=', '+str(x)+'_percentile='+str(all_A1_percentiles[x][i])

            for x in thresh_bump_crits:
                if thresh_bump_crits[x]:
                    working_output+=', '+str(x)+'_thresh_bump_percentile='+str(all_A1_thresh_bump_percentiles[x][i])

            bulk_settings.append(f"Settings_in_Peak_Bulk_Processing_{i} are: ")
            bulk_settings.append(working_output)

            output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Peaks\\"

            # Print a file with all settings for reference
            with open(output_folder+f"Settings_in_Peak_Bulk_Processing_{i}.txt", "w") as output:
                output.write(str(bulk_settings))
    else:
        for i in range(1, max_entries+1):
            bulk_settings.append(filename_no_extension+'-'+str(i))

base_thresh_only = False  # manual override
limit_thresh = False  # manual override

# Grabbing more misc settings and defining trace length
A1_startup_bypass = startup_bypass
queue_size = chunk_size
trace_length = 60

# Handling output .csv and which columns it should contain
col_names = ["latitude", "longitude", "date", "NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)", "WS (m/s)", "WD (degrees)", "WV (m/s)"]
output_cols = ["latitude", "longitude", "date", "NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)", "WS (m/s)", "WD (degrees)", "WV (m/s)",
               "", "NO2 peak (ppb)", "UFP peak (#/cm^3)", "O3 peak (ppb)", "CO peak (ppm)", "CO2 peak (ppm)", "NO peak (ppb)"]
if A1_thresh_dump['no2']:
    output_cols.append('NO2 thresh')
if A1_thresh_dump['ufp']:
    output_cols.append('UFP thresh')
if A1_thresh_dump['o3']:
    output_cols.append('O3 thresh')
if A1_thresh_dump['co']:
    output_cols.append('CO thresh')
if A1_thresh_dump['co2']:
    output_cols.append('CO2 thresh')
if A1_thresh_dump['no']:
    output_cols.append('NO thresh')

# Handling settings in output settings (inside .csv - if desired, default to run other scripts: False)
settings_in_output = {
    'coeff': False,
    'percentile': False,
    'thresh_bump_percentile': False,
    'misc': False
}

# Defining traces
traces = dict(
    no2=deque([], maxlen=trace_length),
    ufp=deque([], maxlen=trace_length),
    o3=deque([], maxlen=trace_length),
    co=deque([], maxlen=trace_length),
    co2=deque([], maxlen=trace_length),
    no=deque([], maxlen=trace_length)
)

# Global counting variables
A1_n = {
    "no2": 0,
    "ufp": 0,
    "o3": 0,
    "co": 0,
    "co2": 0,
    "no": 0
}
ini_current_chunk = 0

# Print warning:
print("NOTE: this script can sometimes take quite some time to run "
      "(typically about 10 seconds for every 5000 rows depending on the user's computer)")

########################################################################################################################
# Declaring functions (helpers) used in the script
########################################################################################################################

# Returns the pollutant value if it's a peak, if not then it returns 0
def ispeak(pollutant):
    global A1_n
    global A1_coeff
    global A1_percentile
    global A1_thresh_bump_percentile
    global traces

    # Compute list of points below our percentile, return 0 if there's an issue
    m = np.percentile(traces[pollutant], A1_percentile[pollutant])
    below_m = []
    for x in traces[pollutant]:
        if x<m:
            below_m.append(x)
    if len(below_m)<2:
        A1_n[pollutant] = 0
        return 0

    # Calculating thresh
    sd = statistics.stdev(below_m)  # stdev will do sample sd and pstdev will do population sd
    thresh = A1_coeff[pollutant] * sd
    if A1_thresh_bump_percentile[pollutant] != 0:
        thresh += np.percentile(traces[pollutant], A1_thresh_bump_percentile[pollutant])

    # Checking appropriate condition
    if traces[pollutant][-1] > thresh:
        if A1_n[pollutant] == 0:
            A1_n[pollutant] += 1
            return traces[pollutant][-1]
        else:
            if traces[pollutant][-1] > (thresh + sd * math.sqrt(A1_n[pollutant])):
                A1_n[pollutant] += 1
                return traces[pollutant][-1]
            else:
                A1_n[pollutant] = 0
                return 0
    else:
        A1_n[pollutant] = 0
        return 0

# Same as ispeak but will return a list of length 2, first value is result of ispeak, second value is the thresh
def ispeakAndThresh(pollutant):
    global A1_n
    global A1_coeff
    global A1_percentile
    global A1_thresh_bump_percentile
    global traces
    global base_thresh_only

    # Compute list of points below our percentile, return 0 if there's an issue
    m = np.percentile(traces[pollutant], A1_percentile[pollutant])
    below_m = []
    for x in traces[pollutant]:
        if x<m:
            below_m.append(x)
    if len(below_m)<2:
        A1_n[pollutant] = 0
        return [0, 0]

    # Calculating thresh
    sd = statistics.stdev(below_m)  # stdev will do sample sd and pstdev will do population sd
    thresh = A1_coeff[pollutant] * sd
    if A1_thresh_bump_percentile[pollutant] != 0:
        thresh += np.percentile(traces[pollutant], A1_thresh_bump_percentile[pollutant])

    # Checking appropriate condition
    if traces[pollutant][-1] > thresh:
        if A1_n[pollutant] == 0:
            A1_n[pollutant] += 1
            return [traces[pollutant][-1], thresh]
        else:
            if traces[pollutant][-1] > (thresh + sd * math.sqrt(A1_n[pollutant])):
                A1_n[pollutant] += 1
                if base_thresh_only:
                    return [traces[pollutant][-1], thresh]
                else:
                    return [traces[pollutant][-1], (thresh + sd * math.sqrt(A1_n[pollutant]))]
            else:
                A1_n[pollutant] = 0
                if base_thresh_only:
                    return [0,thresh]
                else:
                    return [0, (thresh + sd * math.sqrt(A1_n[pollutant]))]
    else:
        A1_n[pollutant] = 0
        return [0, thresh]

# Computes the peak list to be written to the CSV
def compute_peak_list(input_list, pollutant):
    global A1_startup_bypass
    global traces
    global A1_thresh_dump
    global thresh_dict
    output_list=[]

    for i in input_list:
        traces[pollutant].append(i)
        if len(traces[pollutant])<A1_startup_bypass:
            output_list.append(0)
            if A1_thresh_dump[pollutant]:
                thresh_dict[pollutant].append(0)
        else:
            if A1_thresh_dump[pollutant]:
                eval = ispeakAndThresh(pollutant)
                output_list.append(eval[0])
                thresh_dict[pollutant].append(eval[1])
            else:
                output_list.append(ispeak(pollutant))

    return output_list

########################################################################################################################
# Main script
########################################################################################################################

for date in Merged_dates:
    current_chunk = ini_current_chunk
    file_name = f"MERGED_AQ_GPS_{date}.csv"
    full_filename = source_folder+file_name
    out_name = f"Peaks_in_MERGED_AQ_GPS_{date}.csv"
    output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Peaks\\"
    output_folder_fig = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Figures\\Peaks\\"
    output_csv = output_folder+out_name
    print("")
    print("Dataframe currently being processed is: ")
    print(file_name)
    dataframe = pd.read_csv(full_filename)
    print(dataframe)

    if alternative_peaks:
        print("Alternative method for peak detection enabled.")
        pollutants = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
        pollutants_peak = ["Peaks_NO2 (ppb)", "Peaks_UFP (#/cm^3)", "Peaks_O3 (ppb)", "Peaks_CO (ppm)", "Peaks_CO2 (ppm)", "Peaks_NO (ppb)"]
        pol_list = ["NO2_1D", "UFP_1D", "O3_1D", "CO_1D", "CO2_1D", "NO_1D"]
        signal_time = [10, 1, 10, 1, 1, 10]  # instrument update time in seconds
        peaks_list = []

        for pol_number in range(len(pollutants)):
            pol_list[pol_number] = dataframe[pollutants[pol_number]].values  # Transform pollutant column into 1-D array
            pol_height = np.amin(pol_list[pol_number])*1.1  #  Makes the minimum value + 10% as a flat baseline to exclude peaks below it
            pol_distance = signal_time[pol_number]  # The minimum distance between peaks becomes the instrument update time

            # Estimate peaks
            peaks, _ = find_peaks(pol_list[pol_number], height=pol_height, distance=pol_distance)
            # Estimate prominences of each peak
            prominences = peak_prominences(pol_list[pol_number], peaks)[0]
            # Establish a prominence equals to 5% of the maximum value -- it will be used as cut point
            pol_prominence = np.amax(prominences)*0.05
            # Estimate peaks (again, but with prominence and no distance or height)
            peaks, _ = find_peaks(pol_list[pol_number], prominence=pol_prominence)
            # Append the peaks to the list
            peaks_list.append(peaks)

            # Create a new column with peak values, or NaN if no peak was found
            dataframe[f'Peaks_{pollutants[pol_number]}'] = np.where(np.isin(np.arange(len(dataframe)), peaks), dataframe[pollutants[pol_number]], np.nan)

            # Plotting:
            dataframe['date'] = pd.to_datetime(dataframe['date'], format='%Y-%m-%d %H:%M:%S')
            print(f"Plotting peaks for {date} and {pollutants[pol_number]} in {file_name}")
            # Extract the x-axis values from the "date" column
            x = dataframe["date"]
            # Extract the y-axis values from the columns you want to plot
            y1 = dataframe[pollutants[pol_number]]
            y2 = dataframe[pollutants_peak[pol_number]]
            # Create the plot
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.plot(x, y1, label=pollutants[pol_number], color="black")
            ax.scatter(x, y2, label=pollutants_peak[pol_number], color="red", marker='X')
            # Format the x-axis tick labels
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
            # Add a title to the plot
            plt.title(f"Peaks for {pollutants[pol_number]} in {date}")
            concentration_unit = pollutants[pol_number].split("(")[1].split(")")[0]
            plt.suptitle(f"Configuration: scipy.find_peaks(), Height ({concentration_unit}): {pol_height:.1f}, Distance (s): {pol_distance}, Prominence ({concentration_unit}): {pol_prominence:.1f}", fontsize=10, color="gray")
            # Add labels to the x and y axes
            plt.xlabel("Time")
            plt.ylabel(pollutants[pol_number])
            # Add a legend to differentiate the two lines
            plt.legend()
            # Add minor ticks to the y-axis
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            # Add horizontal dashed light-gray lines for every tick
            ax.grid(axis='y', which='both', linestyle='dashed', color='lightgray')
            # Save the plot as an image in the specified folder
            pollutants_newlist = ["NO2 (ppb)", "UFP (counts)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
            filename_without_extension = file_name[:-4]  # Remove the last 4 characters (.csv)
            plt.savefig(output_folder_fig+f'Alternative_Peaks_for_{pollutants_newlist[pol_number]}_in_{filename_without_extension}', dpi=500,
                        transparent=False)
            # Close the plot
            plt.close()

        # Save work:
        print("")
        print(f"Saving: Alternative_Peaks_in_{file_name}")
        dataframe.to_csv(output_folder+f'Alternative_Peaks_in_{file_name}', index=False)

    # Use the main method:
    else:
        if bulk_processing == False:
            while True:
                # Read in the current chunk
                data = pd.read_csv(full_filename, names=col_names, skiprows=(1 + current_chunk * queue_size),
                                   nrows=queue_size)

                # Convert current chunk to lists
                no2_list = data["NO2 (ppb)"].to_list()
                ufp_list = data["UFP (#/cm^3)"].to_list()
                o3_list = data["O3 (ppb)"].to_list()
                co_list = data["CO (ppm)"].to_list()
                co2_list = data["CO2 (ppm)"].to_list()
                no_list = data['NO (ppb)'].to_list()
                ws_list = data['WS (m/s)'].to_list()
                wd_list = data['WD (degrees)'].to_list()
                wv_list = data['WV (m/s)'].to_list()
                lat_list = data['latitude'].to_list()
                long_list = data['longitude'].to_list()
                time_list = data["date"].to_list()

                # Create thresh lists
                thresh_dict = {
                    'no2': [],
                    'ufp': [],
                    'o3': [],
                    'co': [],
                    'co2': [],
                    'no': []
                }

                # Compute peaks for current chunk and save as its own list
                no2_peaks = compute_peak_list(no2_list, 'no2')
                ufp_peaks = compute_peak_list(ufp_list, 'ufp')
                o3_peaks = compute_peak_list(o3_list, 'o3')
                co_peaks = compute_peak_list(co_list, 'co')
                co2_peaks = compute_peak_list(co2_list, 'co2')
                no_peaks = compute_peak_list(no_list, 'no')

                # Limiting thresh if necessary
                if limit_thresh:
                    maxes_dict = {
                        'no2': (max(no2_list)),
                        'ufp': (max(ufp_list)),
                        'o3': (max(o3_list)),
                        'co': (max(co_list)),
                        'co2': (max(co2_list)),
                        'no': (max(no_list))
                    }
                    for pollutant in thresh_dict:
                        for i in range(0, len(thresh_dict[pollutant])):
                            if thresh_dict[pollutant][i] > maxes_dict[pollutant]:
                                thresh_dict[pollutant][i] = maxes_dict[pollutant]

                # Write current chunk to CSV
                with open(output_csv, "a", newline='') as f:
                    w = csv.writer(f)

                    # Write headers if we're on the first chunk
                    if current_chunk == 0:

                        # Writing settings to output file
                        if settings_in_output['coeff']:
                            w.writerow(['[A1_coeff]'])
                            for i in A1_coeff:
                                w.writerow([i + ': ', A1_coeff[i]])
                            w.writerow(['', ''])

                        if settings_in_output['percentile']:
                            w.writerow(['[A1_percentile]'])
                            for i in A1_percentile:
                                w.writerow([i + ': ', A1_percentile[i]])
                            w.writerow(['', ''])

                        if settings_in_output['thresh_bump_percentile']:
                            w.writerow(['[A1_thresh_bump_percentile]'])
                            for i in A1_thresh_bump_percentile:
                                w.writerow([i + ': ', A1_thresh_bump_percentile[i]])
                            w.writerow(['', ''])

                        if settings_in_output['misc']:
                            w.writerow(['startup_bypass: ', A1_startup_bypass])
                            w.writerow(['chunk_size: ', queue_size])
                            w.writerow(['', ''])

                        w.writerow(output_cols)

                    # Write data to CSV
                    for i in range(0, len(no2_list)):
                        row = [lat_list[i], long_list[i], time_list[i], no2_list[i], ufp_list[i], o3_list[i],
                               co_list[i],
                               co2_list[i], no_list[i], ws_list[i], wd_list[i], wv_list[i], ""]

                        # The following lines deal with the rebound effect by setting the peak to 0 if
                        # its value is within 10% of the threshold. This way, false-peaks are not flagged

                        if A1_thresh_dump['no2']:
                            if abs(no2_peaks[i] - thresh_dict['no2'][i]) <= 0.1 * abs(thresh_dict['no2'][i]):
                                no2_peaks[i] = 0
                                row.append(no2_peaks[i])
                            else:
                                row.append(no2_peaks[i])

                        if A1_thresh_dump['ufp']:
                            if abs(ufp_peaks[i] - thresh_dict['ufp'][i]) <= 0.1 * abs(thresh_dict['ufp'][i]):
                                ufp_peaks[i] = 0
                                row.append(ufp_peaks[i])
                            else:
                                row.append(ufp_peaks[i])

                        if A1_thresh_dump['o3']:
                            if abs(o3_peaks[i] - thresh_dict['o3'][i]) <= 0.1 * abs(thresh_dict['o3'][i]):
                                o3_peaks[i] = 0
                                row.append(o3_peaks[i])
                            else:
                                row.append(o3_peaks[i])

                        if A1_thresh_dump['co']:
                            if abs(co_peaks[i] - thresh_dict['co'][i]) <= 0.1 * abs(thresh_dict['co'][i]):
                                co_peaks[i] = 0
                                row.append(co_peaks[i])
                            else:
                                row.append(co_peaks[i])

                        if A1_thresh_dump['co2']:
                            if abs(co2_peaks[i] - thresh_dict['co2'][i]) <= 0.1 * abs(thresh_dict['co2'][i]):
                                co2_peaks[i] = 0
                                row.append(co2_peaks[i])
                            else:
                                row.append(co2_peaks[i])

                        if A1_thresh_dump['no']:
                            if abs(no_peaks[i] - thresh_dict['no'][i]) <= 0.1 * abs(thresh_dict['no'][i]):
                                no_peaks[i] = 0
                                row.append(no_peaks[i])
                            else:
                                row.append(no_peaks[i])

                        row.append(thresh_dict['no2'][i])
                        row.append(thresh_dict['ufp'][i])
                        row.append(thresh_dict['o3'][i])
                        row.append(thresh_dict['co'][i])
                        row.append(thresh_dict['co2'][i])
                        row.append(thresh_dict['no'][i])

                        w.writerow(row)

                # Break loop if we're on the last chunk, otherwise go to next chunk
                if len(no2_list) < queue_size:
                    print("chunk " + str(current_chunk + 1) + " written")
                    break
                else:
                    print("chunk " + str(current_chunk + 1) + " written")
                    current_chunk += 1
        else:
            print('Bulk processing ENABLED')
            for i in range(0, max_entries):

                print('\nComputing peaks ' + str(i + 1) + ' of ' + str(max_entries))

                # Reseting all counters
                traces = dict(
                    no2=deque([], maxlen=trace_length),
                    ufp=deque([], maxlen=trace_length),
                    o3=deque([], maxlen=trace_length),
                    co=deque([], maxlen=trace_length),
                    co2=deque([], maxlen=trace_length),
                    no=deque([], maxlen=trace_length)
                )
                A1_n = {
                    "no2": 0,
                    "ufp": 0,
                    "o3": 0,
                    "co": 0,
                    "co2": 0,
                    "no": 0
                }
                current_chunk = 0

                # Grabbing settings for this run
                A1_coeff = {
                    "no2": all_A1_coeffs['no2'][i],
                    "ufp": all_A1_coeffs['ufp'][i],
                    "o3": all_A1_coeffs['o3'][i],
                    "co": all_A1_coeffs['co'][i],
                    "co2": all_A1_coeffs['co2'][i],
                    "no": all_A1_coeffs['no'][i]
                }
                A1_percentile = {
                    "no2": all_A1_percentiles['no2'][i],
                    "ufp": all_A1_percentiles['ufp'][i],
                    "o3": all_A1_percentiles['o3'][i],
                    "co": all_A1_percentiles['co'][i],
                    "co2": all_A1_percentiles['co2'][i],
                    "no": all_A1_percentiles['no'][i]
                }
                A1_thresh_bump_percentile = {
                    "no2": all_A1_thresh_bump_percentiles['no2'][i],
                    "ufp": all_A1_thresh_bump_percentiles['ufp'][i],
                    "o3": all_A1_thresh_bump_percentiles['o3'][i],
                    "co": all_A1_thresh_bump_percentiles['co'][i],
                    "co2": all_A1_thresh_bump_percentiles['co2'][i],
                    "no": all_A1_thresh_bump_percentiles['no'][i]
                }

                output_name = f"Peaks_in_MERGED_AQ_GPS_{date}_Bulk_processing_{i}_File_{i + 1}_of_{max_entries}.csv"
                output_csv = output_folder + output_name

                while True:
                    # Read in the current chunk
                    data = pd.read_csv(full_filename, names=col_names, skiprows=(1 + current_chunk * queue_size),
                                       nrows=queue_size)

                    # Convert current chunk to lists
                    no2_list = data["NO2 (ppb)"].to_list()
                    ufp_list = data["UFP (#/cm^3)"].to_list()
                    o3_list = data["O3 (ppb)"].to_list()
                    co_list = data["CO (ppm)"].to_list()
                    co2_list = data["CO2 (ppm)"].to_list()
                    no_list = data['NO (ppb)'].to_list()
                    ws_list = data['WS (m/s)'].to_list()
                    wd_list = data['WD (degrees)'].to_list()
                    wv_list = data['WV (m/s)'].to_list()
                    lat_list = data['latitude'].to_list()
                    long_list = data['longitude'].to_list()
                    time_list = data["date"].to_list()

                    # Create thresh lists
                    thresh_dict = {
                        'no2': [],
                        'ufp': [],
                        'o3': [],
                        'co': [],
                        'co2': [],
                        'no': []
                    }

                    # Compute peaks for current chunk and save as its own list
                    no2_peaks = compute_peak_list(no2_list, 'no2')
                    ufp_peaks = compute_peak_list(ufp_list, 'ufp')
                    o3_peaks = compute_peak_list(o3_list, 'o3')
                    co_peaks = compute_peak_list(co_list, 'co')
                    co2_peaks = compute_peak_list(co2_list, 'co2')
                    no_peaks = compute_peak_list(no_list, 'no')

                    # Limiting thresh if necessary
                    if limit_thresh:
                        maxes_dict = {
                            'no2': (max(no2_list)),
                            'ufp': (max(ufp_list)),
                            'o3': (max(o3_list)),
                            'co': (max(co_list)),
                            'co2': (max(co2_list)),
                            'no': (max(no_list))
                        }
                        for pollutant in thresh_dict:
                            for i in range(0, len(thresh_dict[pollutant])):
                                if thresh_dict[pollutant][i] > maxes_dict[pollutant]:
                                    thresh_dict[pollutant][i] = maxes_dict[pollutant]

                    # Write current chunk to CSV
                    with open(output_csv, "a", newline='') as f:
                        w = csv.writer(f)

                        # Write headers if we're on the first chunk
                        if current_chunk == 0:

                            # Writing settings to output file
                            if settings_in_output['coeff']:
                                w.writerow(['[A1_coeff]'])
                                for i in A1_coeff:
                                    w.writerow([i + ': ', A1_coeff[i]])
                                w.writerow(['', ''])

                            if settings_in_output['percentile']:
                                w.writerow(['[A1_percentile]'])
                                for i in A1_percentile:
                                    w.writerow([i + ': ', A1_percentile[i]])
                                w.writerow(['', ''])

                            if settings_in_output['thresh_bump_percentile']:
                                w.writerow(['[A1_thresh_bump_percentile]'])
                                for i in A1_thresh_bump_percentile:
                                    w.writerow([i + ': ', A1_thresh_bump_percentile[i]])
                                w.writerow(['', ''])

                            if settings_in_output['misc']:
                                w.writerow(['startup_bypass: ', A1_startup_bypass])
                                w.writerow(['chunk_size: ', queue_size])
                                w.writerow(['', ''])

                            w.writerow(output_cols)

                        # Write data to CSV
                        for i in range(0, len(no2_list)):
                            row = [lat_list[i], long_list[i], time_list[i], no2_list[i], ufp_list[i], o3_list[i],
                                   co_list[i], co2_list[i], no_list[i], ws_list[i], wd_list[i], wv_list[i], ""]

                            # The following lines deal with the rebound effect by setting the peak to 0 if
                            # its value is within 10% of the threshold. This way, false-peaks are not flagged

                            if A1_thresh_dump['no2']:
                                if abs(no2_peaks[i] - thresh_dict['no2'][i]) <= 0.1 * (thresh_dict['no2'][i]):
                                    no2_peaks[i] = 0
                                    row.append(no2_peaks[i])
                                else:
                                    row.append(no2_peaks[i])

                            if A1_thresh_dump['ufp']:
                                if abs(ufp_peaks[i] - thresh_dict['ufp'][i]) <= 0.1 * abs(thresh_dict['ufp'][i]):
                                    ufp_peaks[i] = 0
                                    row.append(ufp_peaks[i])
                                else:
                                    row.append(ufp_peaks[i])

                            if A1_thresh_dump['o3']:
                                if abs(o3_peaks[i] - thresh_dict['o3'][i]) <= 0.1 * abs(thresh_dict['o3'][i]):
                                    o3_peaks[i] = 0
                                    row.append(o3_peaks[i])
                                else:
                                    row.append(o3_peaks[i])

                            if A1_thresh_dump['co']:
                                if abs(co_peaks[i] - thresh_dict['co'][i]) <= 0.1 * abs(thresh_dict['co'][i]):
                                    co_peaks[i] = 0
                                    row.append(co_peaks[i])
                                else:
                                    row.append(co_peaks[i])

                            if A1_thresh_dump['co2']:
                                if abs(co2_peaks[i] - thresh_dict['co2'][i]) <= 0.1 * abs(thresh_dict['co2'][i]):
                                    co2_peaks[i] = 0
                                    row.append(co2_peaks[i])
                                else:
                                    row.append(co2_peaks[i])

                            if A1_thresh_dump['no']:
                                if abs(no_peaks[i] - thresh_dict['no'][i]) <= 0.1 * abs(thresh_dict['no'][i]):
                                    no_peaks[i] = 0
                                    row.append(no_peaks[i])
                                else:
                                    row.append(no_peaks[i])

                            row.append(thresh_dict['no2'][i])
                            row.append(thresh_dict['ufp'][i])
                            row.append(thresh_dict['o3'][i])
                            row.append(thresh_dict['co'][i])
                            row.append(thresh_dict['co2'][i])
                            row.append(thresh_dict['no'][i])

                            w.writerow(row)

                    # break loop if we're on the last chunk, otherwise go to next chunk
                    if len(no2_list) < queue_size:
                        print("chunk " + str(current_chunk + 1) + " written")
                        break
                    else:
                        print("chunk " + str(current_chunk + 1) + " written")
                        current_chunk += 1


# Get Merged AQ + GPS file names and dates
source_folder_peaks = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Peaks\\"
source_folder_baselines = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Baselines\\"
file_names_peaks = os.listdir(source_folder_peaks)
file_names_baselines = os.listdir(source_folder_baselines)

# Baseline in Merged AQ + GPS data (file names):
Baseline_merged_names = [file_name for file_name in file_names_baselines if
                         re.match(r"Best_Baseline_Prediction", file_name)]
Baseline_merged_names = [os.path.splitext(file_name)[0] for file_name in Baseline_merged_names]  # Get rid of the .csv
# Peaks in Merged AQ + GPS data (file names):
Peak_merged_names = [file_name for file_name in file_names_peaks if re.match(r"Peaks_in_MERGED_AQ_GPS", file_name)]
Peak_merged_names = [os.path.splitext(file_name)[0] for file_name in Peak_merged_names]  # Get rid of the .csv

print("")
print("Peak files created. Analyzing...")

# Dates:
date_pattern = r"\d{4}_\d{2}_\d{2}"  # Pattern to match the date in the format YYYY_MM_DD
date_list_baselines = []
date_list_peaks = []

for string in Baseline_merged_names:
    match = re.search(date_pattern, string)
    if match:
        date = match.group()  # Extract the matched date
        date_list_baselines.append(date)  # Add the date to the list

for string in Peak_merged_names:
    match = re.search(date_pattern, string)
    if match:
        date = match.group()  # Extract the matched date
        date_list_peaks.append(date)  # Add the date to the list

if plotting_from_baseline:
    date_list = list(set(date_list_baselines))
    print("Dates in those files are: ", date_list)
    print("")
    print("Peaks from baseline subtraction plotting enabled")
    print("")
    output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Figures\\Peaks\\"

    for date in date_list:
        # Prefix string
        prefix = f"Signal_After_Best_Baseline_Subtraction_{date}"

        # Get the list of filenames that start with the specified prefix
        matching_files = [filename for filename in os.listdir(source_folder_baselines) if filename.startswith(prefix)]

        # Read each matching CSV file into a dataframe
        for filename in matching_files:
            file_path = os.path.join(source_folder_baselines, filename)
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
            pollutants = ["NO2 signal (ppb)", "UFP signal (#/cm^3)", "O3 signal (ppb)", "CO signal (ppm)",
                          "CO2 signal (ppm)", "NO signal (ppb)"]

            for pollutant_num in range(len(pollutants)):
                print(
                    f"Plotting peaks after baseline subtraction for {date} and {pollutants[pollutant_num]} in {filename}")
                # Extract the x-axis values from the "date" column
                x = data["date"]
                # Extract the y-axis values from the columns you want to plot
                y = data[pollutants[pollutant_num]]
                # Create the plot
                fig, ax = plt.subplots(figsize=(16, 5))
                ax.plot(x, y, label=pollutants[pollutant_num], color="black")
                # Format the x-axis tick labels
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                # Add a title to the plot
                plt.title(f"{pollutants[pollutant_num]} in {date} after baseline subtraction")
                # Add labels to the x and y axes
                plt.xlabel("Time")
                plt.ylabel(pollutants[pollutant_num])
                # Add a legend to differentiate the two lines
                plt.legend()
                # Add minor ticks to the y-axis
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                # Add horizontal dashed light-gray lines for every tick
                plt.grid(axis='y', which='both', linestyle='dashed', color='lightgray')
                # Save the plot as an image in the specified folder
                pollutants_newlist = ["NO2 (ppb)", "UFP (counts)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
                plt.savefig(output_folder + f"Baseline_subtracted_peaks_for_" + date + "_" + pollutants_newlist[pollutant_num],
                            dpi=500, transparent=False)
                # Close the plot
                plt.close()
else:
    print("Plotting peaks after baseline subtraction not enabled")

if alternative_peaks:
    # Get a list of all "Settings_in_Peak_Bulk_Processing" files in the folder
    output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Peaks\\"
    file_list = os.listdir(output_folder)

    # Iterate over each file in the folder
    for file_name in file_list:
        if file_name.endswith('.txt') and file_name.startswith('Settings_in_Peak_Bulk_Processing_'):
            file_path = os.path.join(output_folder, file_name)  # Get the full file path
            os.remove(file_path)  # Delete the file
    print("Run ended.")
else:
    if plotting_results:
        date_list = list(set(date_list_peaks))
        print("Dates in those files are: ", date_list)
        print("")
        print("Peaks from peak detection algorithm plotting enabled")
        print("")
        output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Figures\\Peaks\\"

        for date in date_list:
            # Prefix string
            prefix = f"Peaks_in_MERGED_AQ_GPS_{date}"

            # Get the list of filenames that start with the specified prefix
            matching_files = [filename for filename in os.listdir(source_folder_peaks) if filename.startswith(prefix)]

            # Read each matching CSV file into a dataframe
            for filename in matching_files:
                file_path = os.path.join(source_folder_peaks, filename)
                data = pd.read_csv(file_path, low_memory=False)
                data.replace(0, np.nan, inplace=True)  # Necessary for better plots to set zeros to NaN
                data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')

                pollutants = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
                pollutants_peak = ["NO2 peak (ppb)", "UFP peak (#/cm^3)", "O3 peak (ppb)",
                               "CO peak (ppm)",
                               "CO2 peak (ppm)", "NO peak (ppb)"]
                pollutants_thresh = ["NO2 thresh", "UFP thresh", "O3 thresh",
                                 "CO thresh",
                                 "CO2 thresh", "NO thresh"]

                for pollutant_num in range(len(pollutants)):
                    print(f"Plotting peaks for {date} and {pollutants[pollutant_num]} in {filename}")
                    # Extract the x-axis values from the "date" column
                    x = data["date"]
                    # Extract the y-axis values from the columns you want to plot
                    y1 = data[pollutants[pollutant_num]]
                    y2 = data[pollutants_peak[pollutant_num]]
                    # y3 = data[pollutants_thresh[pollutant_num]]
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(16, 5))
                    ax.plot(x, y1, label=pollutants[pollutant_num], color="black")
                    ax.scatter(x, y2, label=pollutants_peak[pollutant_num], color="red", marker='X')
                    # ax.plot(x, y3, label=pollutants_thresh[pollutant_num], color="gray")
                    # Format the x-axis tick labels
                    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                    # Add a title to the plot
                    plt.title(f"Peaks for {pollutants[pollutant_num]} in {date}")
                    plt.suptitle(f"Configuration: {filename}", fontsize=10, color="gray")
                    # Add labels to the x and y axes
                    plt.xlabel("Time")
                    plt.ylabel(pollutants[pollutant_num])
                    # Add a legend to differentiate the two lines
                    plt.legend()
                    # Add minor ticks to the y-axis
                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                    # Add horizontal dashed light-gray lines for every tick
                    ax.grid(axis='y', which='both', linestyle='dashed', color='lightgray')
                    # Save the plot as an image in the specified folder
                    start_index = filename.find(f"Peaks_in_MERGED_AQ_GPS_{date}")
                    end_index = filename.find(".csv")
                    middle_part = filename[start_index:end_index]
                    pollutants_newlist = ["NO2 (ppb)", "UFP (counts)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
                    plt.savefig(output_folder + pollutants_newlist[pollutant_num] + "_" + date + "_" + middle_part, dpi=500,
                            transparent=False)
                    # Close the plot
                    plt.close()
    else:
        print("Plotting ALL peaks not enabled")

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

