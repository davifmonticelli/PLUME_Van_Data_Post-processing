# This script was created to automatically post-process P.L.U.M.E. Van data after multiple days of sampling
# Function: From merged (Sensor transcript + GPS), it creates baselines and selects the best baseline across pollutants
# Authors: Chris Kelly and Davi de Ferreyro Monticelli, iREACH group (University of British Columbia)
# Date: 2023-06-20
# Version: 1.0.0

# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "Pre-processing_PLUME_Data.py" PRIOR
# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "auto_merge_data.py" PRIOR

import os
import io
import re
import math
import csv
import sys
import pandas as pd
from collections import deque
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
file_path_txt = f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Baselines\\Baseline_console_output.txt'

# Open the file in write mode and create the Tee object
output_file_txt = open(file_path_txt, 'w')
tee = Tee(output_file_txt)

# Redirect the standard output to the Tee object
sys.stdout = tee

########################################################################################################################
start_time = time.time()
start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print("Start time of run: ", start_time_formatted)

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

# Baseline settings:
# Please refer to the Dashboard paper and supporting information found at: https://doi.org/10.1016/j.envsoft.2022.105600
# to understand how and when to adjust these settings
window_size = [3]
smoothing_index = [15]
chunk_size = [3000]
interlace_chunks = True
folder_directory = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\"
input_filenames = [file_name for file_name in file_names if re.match(r"MERGED_AQ_GPS_\d{4}_\d{2}_\d{2}.csv", file_name)]
output_filenames = ['Baseline_' + item for item in input_filenames]
settings_in_name = True
print("Input file names are: ", input_filenames)
print("Output file names will start with: ", output_filenames)

print("NOTE: This script takes considerable amount of time if bulk processing is enabled. "
      "For a good baseline performance but saving time, set it to False.")

# Baseline bulk processing settings:
# Please refer to the Dashboard paper and supporting information found at: https://doi.org/10.1016/j.envsoft.2022.105600
# to understand how and when to adjust these settings.
#
# Briefly:
# override previous settings, and you can test different configurations in one run
enable_bulk_processing = False
# We recommend changing only the smoothing indexes.
# Furthermore, after numerous trials and visual inspection 15 appears to be the best overall setting
window_sizes = [3]  # Recommended to be 3, 4 or 5
smoothing_indexes = [i for i in range(1, 61)]  # Iterates from 1 to 60 so it takes time (2h to 4h per sampling day with 20000 datapoitns)

# Save all baselines files created?
# Each day will have X baselines computed if bulk processing is enabled, thus X .csv files
# If this is set to False, at the end of the run, the script will delete these files and keep only the best baseline .csv
save_all_baselines = True

# Baseline comparison settings:
# This is a NEW FEATURE, where it compares the baselines generated through bulk processing and spills the best-one.
# Best-one = baseline that isolates the peaks better in the time series.
# It uses the PSNR (Peak Signal-to-Noise-Ratio) to indicate the ability to separate the peaks from the noise.
enable_baseline_comparison = True
plotting_psnr_analysis = True  # To inspect the analysis

# Baseline abstraction (peaks isolation):
# if enabled, it will use the previous created file ( the best baseline ) and remove the baseline from signal for each
# pollutant, thus generating a file with only peaks
enable_baseline_subtraction = True

# Plotting baselines:
# Plotting from all files:
plotting_baselines_allfiles = False
# Plotting from the best-baseline files:
plotting_baselines_bestfiles = True

#######################################################################################################################
# Functions required to run the script:
#######################################################################################################################

# Used for grabbing bulk processing
def full_string_to_int_list(string_in):
    if isinstance(string_in, str):
        # remove all spaces from string_in
        string_in = string_in.replace(" ", "")

        string_in += ','
        output=[]
        adding_to_result=''

        for i in range(0,len(string_in)):
            if string_in[i] == ',':
                output.append(int(adding_to_result))
                adding_to_result=''
            else:
                adding_to_result += string_in[i]

    elif isinstance(string_in, list):
        output = string_in

    return output

# Fills in settings for smaller bulk processing list
def fill_smaller_bulk(smaller,target_length):
    while len(smaller) < target_length:
        smaller.append(smaller[-1])
    return smaller

# Helper function for interlaced chunks
def overwrite_last_half(og_list, more_list):
    halfway_point = int(0.5*len(og_list))
    for i in range(halfway_point,len(og_list)):
        og_list[i] = more_list[i]
    return og_list

# Helper function for interlaced chunks
def overwrite_first_half(current_list, more_list):
    halfway_of_more_list = int(0.5*len(more_list))
    k=0
    for i in range(halfway_of_more_list, int(1.5*halfway_of_more_list)):
        current_list[k] = more_list[i]
        k+=1
    return current_list

# Helper function to interpolate the "-" characters into numbers
def interpolate(input_pass):
    pair = deque([input_pass[0]], maxlen=2)
    dashes_between = 0

    for i in range(1,len(input_pass)):
        if type(input_pass[i]) == int or type(input_pass[i]) == float:

            #bring in the next number
            pair.append(input_pass[i])



            #compute slope increment
            slope = (pair[1] - pair[0]) / (dashes_between+1)

            #fill values
            inc = 1
            for f in range(i-dashes_between, i):
                input_pass[f] = round( (input_pass[i - dashes_between - 1 ] + (inc * slope)),2 )
                inc += 1


            dashes_between = 0
        else:
            dashes_between += 1

    return input_pass

# Helper function to make our passes the same length of data_chunk
def fill_trailing_data(input_pass, data_chunk):
    while len(input_pass) < (len(data_chunk) - 1):
        input_pass.append("-")
    if len(input_pass) == (len(data_chunk) - 1):
        input_pass.append(data_chunk[-1])
    else:
        input_pass[-1] = data_chunk[-1]
    return input_pass

# Helper function to compute our baseline assuming smoothing index = 1
def compute_baseline_no_smoothing(data_chunk, window_size):
    if isinstance(window_size, list):
        # Perform action for a list
        window_size = window_size[0]
        # Additional code for handling lists
    elif isinstance(window_size, int):
        # Perform action for an integer
        window_size = window_size
        # Additional code for handling integers

    #lists storing each pass, these will be averaged out later
    pass1=[data_chunk[0]]
    pass2=[data_chunk[0]]
    pass3=[data_chunk[0]]

    #iterate through current data_chunk, starting at SECOND division of interval and iterating one interval at a time
    for i in range(window_size, len(data_chunk), window_size):
        working_window = data_chunk[(i-window_size+1):(i+1)]
        window_min = min(working_window)
        window_min_index = working_window.index(window_min)
        working_window = ["-"]*window_size
        working_window[window_min_index] = window_min
        pass1+=working_window
    #calling fill_trailing_data function to add trailing data to pass1
    pass1 = fill_trailing_data(pass1, data_chunk)


    #defining our offset and adding "-" characters to the start of passes 2 and 3
    offset = math.floor(window_size/3)
    for i in range(0,(offset)):
        pass2.append("-")
    for i in range(0,(offset*2)):
        pass3.append("-")

    #repeating the same process for passes 2 and 3, with an offset in start window
    for i in range(window_size + offset, len(data_chunk), window_size):
        working_window = data_chunk[(i-window_size+1):(i+1)]
        window_min = min(working_window)
        window_min_index = working_window.index(window_min)
        working_window = ["-"]*window_size
        working_window[window_min_index] = window_min
        pass2+=working_window
    pass2 = fill_trailing_data(pass2, data_chunk)
    for i in range(window_size + (2*offset), len(data_chunk), window_size):
        working_window = data_chunk[(i-window_size+1):(i+1)]
        window_min = min(working_window)
        window_min_index = working_window.index(window_min)
        working_window = ["-"]*window_size
        working_window[window_min_index] = window_min
        pass3+=working_window
    pass3 = fill_trailing_data(pass3, data_chunk)


    #call the interpolate function for each of the passes
    pass1 = interpolate(pass1)
    pass2 = interpolate(pass2)
    pass3 = interpolate(pass3)

    average_of_passes=[]

    for i in range(0,len(pass1)):
        average_of_passes.append  ( round(    ( (pass1[i] + pass2[i] + pass3[i]) / 3.0 ),3 ))


    #special_return = [[pass1],[pass2],[pass3],[average_of_passes]] #for debugging purposes
    #return special_return

    return average_of_passes

# Function that computes the final baseline, uses all of the above helper functions
def compute_baseline(data_chunk, window_size, smoothing):
    if enable_bulk_processing:
        size = window_sizes[0]
        output_list = compute_baseline_no_smoothing(data_chunk, size)
    else:
        size = window_size
        output_list = compute_baseline_no_smoothing(data_chunk, size)

    if smoothing > 1:
        for i in range(2, smoothing+1):
            average_with = compute_baseline_no_smoothing(data_chunk, size*i)
            for f in range(0,len(output_list)):
                output_list[f] = (output_list[f] + average_with[f])/2.0

    # Set all values to actual if the baseline value is above actual and round to 6 places
    # De-activated considering the atmospheric chemistry of the pollutants
    #for i in range(0, len(output_list)):
    #    if output_list[i] > data_chunk[i]:
    #        output_list[i] = data_chunk[i]
    #    output_list[i] = round(output_list[i],6)

    return output_list

# Handling output filename
def output_file(number):
    folder_directory = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Baselines\\"
    output_csv = folder_directory+output_filenames[number]
    if (include_settings_in_filename):
        filename_no_extension = ''
        for i in range(0, len(output_csv)):
            if output_csv[(i):(i + 4)].lower() == '.csv':
                filename_no_extension = output_csv[0:i]
        output_csv = filename_no_extension + ", window_size = " + str(
            setting_window_size) + ', smoothing_index = ' + str(setting_smoothing) + ', chunk_size=' + str(queue_size)
        if interlace_chunks:
            output_csv += ", interlaced chunks.csv"
        else:
            output_csv += ', not interlaced.csv'

    return output_csv

def find_closest_index(lst):
    target_value = -0.5
    closest_value = min(lst, key=lambda x: abs(x - target_value))
    closest_index = lst.index(closest_value)
    return closest_index

########################################################################################################################
# GLOBAL SETTINGS
########################################################################################################################

bulk_processing = enable_bulk_processing

# Dealing with bulk processing specific settings
if bulk_processing:
    include_settings_in_filename = True
    # Get the parameter with repetitive values and fixed one and create new list
    if window_sizes > smoothing_indexes and len(smoothing_indexes) == 1:
        smoothing_indexes = [smoothing_indexes] * len(window_sizes)
    elif smoothing_indexes > window_sizes and len(window_sizes) == 1:
        window_sizes = [window_sizes] * len(smoothing_indexes)
    else:
        window_sizes = window_sizes
        smoothing_indexes = smoothing_indexes

    all_window_sizes = window_sizes
    setting_window_size = all_window_sizes[0]
    all_smoothing_indexes = full_string_to_int_list(smoothing_indexes)
    setting_smoothing = all_smoothing_indexes[0]

    if len(all_window_sizes) >= len(all_smoothing_indexes):
        runs = len(all_window_sizes)
        all_smoothing_indexes = fill_smaller_bulk(all_smoothing_indexes,runs)
    else:
        runs = len(all_smoothing_indexes)
        all_window_sizes = fill_smaller_bulk(all_window_sizes,runs)
else:
    setting_window_size = window_size[0]
    setting_smoothing = smoothing_index[0]
    include_settings_in_filename = settings_in_name

queue_size = chunk_size[0]  # effectively "chunk size"
is_formatted = True  # manual override

col_names = ["latitude", "longitude", "date", "NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)", "WS (m/s)", "WD (degrees)", "WV (m/s)"]
output_cols = ["latitude", "longitude", "date", "NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)", "WS (m/s)", "WD (degrees)", "WV (m/s)",
               "", "NO2 baseline (ppb)", "UFP baseline (#/cm^3)", "O3 baseline (ppb)", "CO baseline (ppm)", "CO2 baseline (ppm)","NO baseline (ppb)"]
settings_in_output = False  # manual override

ini_current_chunk = 0  # Global counter, DO NOT CHANGE, KEEP IT SET AT 0

########################################################################################################################
# MAIN SCRIPT
########################################################################################################################

times_to_run = len(Merged_names)

# Non-bulk processing:
for number in range(times_to_run):
    current_chunk = ini_current_chunk
    print("")
    print("Processing dataframe:", input_filenames[number])
    dataframe = pd.read_csv(folder_directory + input_filenames[number], names=col_names, skiprows=(1 + current_chunk * queue_size), nrows=queue_size)
    print(dataframe)

    if not bulk_processing:
        if interlace_chunks:
            while True:
                # Read in the current chunk
                data = pd.read_csv(folder_directory + input_filenames[number], names=col_names, skiprows=(1 + current_chunk * queue_size),
                                   nrows=queue_size)

                # Convert current chunk to lists
                no2_list = data["NO2 (ppb)"].to_list()
                ufp_list = data["UFP (#/cm^3)"].to_list()
                o3_list = data["O3 (ppb)"].to_list()
                co_list = data["CO (ppm)"].to_list()
                co2_list = data["CO2 (ppm)"].to_list()
                no_list = data["NO (ppb)"].to_list()
                ws_list = data["WS (m/s)"].to_list()
                wd_list = data["WD (degrees)"].to_list()
                wv_list = data["WV (m/s)"].to_list()
                time_list = data["date"].to_list()
                latitude_list = data["latitude"].to_list()
                longitude_list = data["longitude"].to_list()

                # Compute baseline for current chunk and save as its own list
                no2_baseline = compute_baseline(no2_list, setting_window_size, setting_smoothing)
                ufp_baseline = compute_baseline(ufp_list, setting_window_size, setting_smoothing)
                o3_baseline = compute_baseline(o3_list, setting_window_size, setting_smoothing)
                co_baseline = compute_baseline(co_list, setting_window_size, setting_smoothing)
                co2_baseline = compute_baseline(co2_list, setting_window_size, setting_smoothing)
                no_baseline = compute_baseline(no_list, setting_window_size, setting_smoothing)

                # Check if there's data from previous chunk that we can use for interlacing
                if (current_chunk != 0):
                    if more_lists_full:
                        no2_baseline = overwrite_first_half(no2_baseline, no2_baseline_more)
                        ufp_baseline = overwrite_first_half(ufp_baseline, ufp_baseline_more)
                        o3_baseline = overwrite_first_half(o3_baseline, o3_baseline_more)
                        co_baseline = overwrite_first_half(co_baseline, co_baseline_more)
                        co2_baseline = overwrite_first_half(co2_baseline, co2_baseline_more)
                        no_baseline = overwrite_first_half(no_baseline, no_baseline_more)

                # Check if there's data ahead that we can use for interlacing
                if len(no2_list) == queue_size:
                    # Read in current chunk with the first half of the next chunk
                    data_more = pd.read_csv(folder_directory + input_filenames[number], names=col_names, skiprows=(1 + current_chunk * queue_size),
                                            nrows=int(2 * queue_size))

                    # Save this increased chunk to new lists
                    no2_list_more = data_more["NO2 (ppb)"].to_list()
                    ufp_list_more = data_more["UFP (#/cm^3)"].to_list()
                    o3_list_more = data_more["O3 (ppb)"].to_list()
                    co_list_more = data_more["CO (ppm)"].to_list()
                    co2_list_more = data_more["CO2 (ppm)"].to_list()
                    no_list_more = data_more['NO (ppb)'].to_list()

                    # Label current more lists as full or not
                    if len(no2_list_more) == (2 * queue_size):
                        more_lists_full = True
                    else:
                        more_lists_full = False

                    # Compute baseline of increased chunks
                    no2_baseline_more = compute_baseline(no2_list_more, setting_window_size, setting_smoothing)
                    ufp_baseline_more = compute_baseline(ufp_list_more, setting_window_size, setting_smoothing)
                    o3_baseline_more = compute_baseline(o3_list_more, setting_window_size, setting_smoothing)
                    co_baseline_more = compute_baseline(co_list_more, setting_window_size, setting_smoothing)
                    co2_baseline_more = compute_baseline(co2_list_more, setting_window_size, setting_smoothing)
                    no_baseline_more = compute_baseline(no_list_more, setting_window_size, setting_smoothing)

                    # Override second half of baseline lists with the corresponding value in its corresponding baseline_more list
                    no2_baseline = overwrite_last_half(no2_baseline, no2_baseline_more)
                    ufp_baseline = overwrite_last_half(ufp_baseline, ufp_baseline_more)
                    o3_baseline = overwrite_last_half(o3_baseline, o3_baseline_more)
                    co_baseline = overwrite_last_half(co_baseline, co_baseline_more)
                    co2_baseline = overwrite_last_half(co2_baseline, co2_baseline_more)
                    no_baseline = overwrite_last_half(no_baseline, no_baseline_more)

                # Handling output filename
                output_csv = output_file(number)

                with open(output_csv, "a", newline='') as f:
                    w = csv.writer(f)

                    if current_chunk == 0:

                        # Write settings to output
                        if settings_in_output:
                            w.writerow(['window_size: ', setting_window_size])
                            w.writerow(['smoothing_index: ', setting_smoothing])
                            w.writerow(['chunk_size: ', queue_size])
                            w.writerow(['interlace_chunks: ', interlace_chunks])
                            w.writerow(['', ''])

                        w.writerow(output_cols)

                    for i in range(0, len(no2_list)):
                        w.writerow(
                            [latitude_list[i], longitude_list[i], time_list[i], no2_list[i], ufp_list[i], o3_list[i], co_list[i], co2_list[i],
                             no_list[i], ws_list[i], wd_list[i], wv_list[i], "", no2_baseline[i], ufp_baseline[i], o3_baseline[i],
                             co_baseline[i], co2_baseline[i], no_baseline[i]])

                # Break loop if we're on the last chunk, otherwise go to next chunk
                if len(no2_list) < queue_size:
                    print("Baseline chunk " + str(current_chunk + 1) + " written")
                    break
                else:
                    print("Baseline chunk " + str(current_chunk + 1) + " written")
                    current_chunk += 1

        else:
            while True:
                # Read in the current chunk
                data = pd.read_csv(folder_directory + input_filenames[number], names=col_names, skiprows=(1 + current_chunk * queue_size),
                                   nrows=queue_size)

                # Convert current chunk to lists
                no2_list = data["NO2 (ppb)"].to_list()
                ufp_list = data["UFP (#/cm^3)"].to_list()
                o3_list = data["O3 (ppb)"].to_list()
                co_list = data["CO (ppm)"].to_list()
                co2_list = data["CO2 (ppm)"].to_list()
                no_list = data["NO (ppb)"].to_list()
                ws_list = data["WS (m/s)"].to_list()
                wd_list = data["WD (degrees)"].to_list()
                wv_list = data["WV (m/s)"].to_list()
                time_list = data["date"].to_list()
                latitude_list = data["latitude"].to_list()
                longitude_list = data["longitude"].to_list()

                # Compute baseline for current chunk and save as its own list
                no2_baseline = compute_baseline(no2_list, setting_window_size, setting_smoothing)
                ufp_baseline = compute_baseline(ufp_list, setting_window_size, setting_smoothing)
                o3_baseline = compute_baseline(o3_list, setting_window_size, setting_smoothing)
                co_baseline = compute_baseline(co_list, setting_window_size, setting_smoothing)
                co2_baseline = compute_baseline(co2_list, setting_window_size, setting_smoothing)
                no_baseline = compute_baseline(no_list, setting_window_size, setting_smoothing)

                output_csv = output_file(number)

                with open(output_csv, "a", newline='') as f:
                    w = csv.writer(f)

                    if current_chunk == 0:

                        # Write settings to output
                        if settings_in_output:
                            w.writerow(['window_size: ', setting_window_size])
                            w.writerow(['smoothing_index: ', setting_smoothing])
                            w.writerow(['chunk_size: ', queue_size])
                            w.writerow(['interlace_chunks: ', interlace_chunks])
                            w.writerow(['', ''])

                        w.writerow(output_cols)

                    for i in range(0, len(no2_list)):
                        w.writerow([latitude_list[i], longitude_list[i], time_list[i], no2_list[i], ufp_list[i],
                                 o3_list[i], co_list[i], co2_list[i],
                                 no_list[i], ws_list[i], wd_list[i], wv_list[i], "", no2_baseline[i], ufp_baseline[i],
                                 o3_baseline[i],
                                 co_baseline[i], co2_baseline[i], no_baseline[i]])

                # Break loop if we're on the last chunk, otherwise go to next chunk
                if len(no2_list) < queue_size:
                    print("Baseline chunk " + str(current_chunk + 1) + " written")
                    break
                else:
                    print("Baseline chunk " + str(current_chunk + 1) + " written")
                    current_chunk += 1

    else:
        # Creating all output csv filenames
        output_csv = output_file(number)
        output_names = [output_csv]

        for i in range(1, runs):
            next_name = output_csv + ", window_size = " + str(
                all_window_sizes[i]) + ', smoothing_index = ' + str(all_smoothing_indexes[i]) + ', chunk_size=' + str(
                queue_size)
            if interlace_chunks:
                next_name += ", interlaced chunks.csv"
            else:
                next_name += ', not interlaced.csv'
            output_names.append(next_name)

        print('Bulk processing ENABLED')

        for run in range(0, runs):
            current_chunk = 0
            setting_window_size = all_window_sizes[run]
            setting_smoothing = all_smoothing_indexes[run]
            output_csv = output_names[run]
            print('\nComputing baseline ' + str(run + 1) + ' of ' + str(runs) + ', window_size = ' + str(
                setting_window_size) + ', smoothing_index = ' + str(setting_smoothing))

            if interlace_chunks:
                while True:
                    # Read in the current chunk
                    data = pd.read_csv(folder_directory + input_filenames[number], names=col_names, skiprows=(1 + current_chunk * queue_size), nrows=queue_size)

                    # Convert current chunk to lists
                    no2_list = data["NO2 (ppb)"].to_list()
                    ufp_list = data["UFP (#/cm^3)"].to_list()
                    o3_list = data["O3 (ppb)"].to_list()
                    co_list = data["CO (ppm)"].to_list()
                    co2_list = data["CO2 (ppm)"].to_list()
                    no_list = data["NO (ppb)"].to_list()
                    ws_list = data["WS (m/s)"].to_list()
                    wd_list = data["WD (degrees)"].to_list()
                    wv_list = data["WV (m/s)"].to_list()
                    time_list = data["date"].to_list()
                    latitude_list = data["latitude"].to_list()
                    longitude_list = data["longitude"].to_list()

                    # Compute baseline for current chunk and save as its own list
                    no2_baseline = compute_baseline(no2_list, setting_window_size, setting_smoothing)
                    ufp_baseline = compute_baseline(ufp_list, setting_window_size, setting_smoothing)
                    o3_baseline = compute_baseline(o3_list, setting_window_size, setting_smoothing)
                    co_baseline = compute_baseline(co_list, setting_window_size, setting_smoothing)
                    co2_baseline = compute_baseline(co2_list, setting_window_size, setting_smoothing)
                    no_baseline = compute_baseline(no_list, setting_window_size, setting_smoothing)

                    # Check if there's data from previous chunk that we can use for interlacing
                    if (current_chunk != 0):
                        if more_lists_full:
                            no2_baseline = overwrite_first_half(no2_baseline, no2_baseline_more)
                            ufp_baseline = overwrite_first_half(ufp_baseline, ufp_baseline_more)
                            o3_baseline = overwrite_first_half(o3_baseline, o3_baseline_more)
                            co_baseline = overwrite_first_half(co_baseline, co_baseline_more)
                            co2_baseline = overwrite_first_half(co2_baseline, co2_baseline_more)
                            no_baseline = overwrite_first_half(no_baseline, no_baseline_more)

                    # Check if there's data ahead that we can use for interlacing
                    if len(no2_list) == queue_size:
                        # Read in current chunk with the first half of the next chunk
                        data_more = pd.read_csv(folder_directory+str(input_filenames[number]), names=col_names, skiprows=(1 + current_chunk * queue_size),
                                                nrows=int(2 * queue_size))

                        # Save this increased chunk to new lists
                        no2_list_more = data_more["NO2 (ppb)"].to_list()
                        ufp_list_more = data_more["UFP (#/cm^3)"].to_list()
                        o3_list_more = data_more["O3 (ppb)"].to_list()
                        co_list_more = data_more["CO (ppm)"].to_list()
                        co2_list_more = data_more["CO2 (ppm)"].to_list()
                        no_list_more = data_more['NO (ppb)'].to_list()

                        # Label current more lists as full or not
                        if len(no2_list_more) == (2 * queue_size):
                            more_lists_full = True
                        else:
                            more_lists_full = False

                        # Compute baseline of increased chunks
                        no2_baseline_more = compute_baseline(no2_list_more, setting_window_size, setting_smoothing)
                        ufp_baseline_more = compute_baseline(ufp_list_more, setting_window_size, setting_smoothing)
                        o3_baseline_more = compute_baseline(o3_list_more, setting_window_size, setting_smoothing)
                        co_baseline_more = compute_baseline(co_list_more, setting_window_size, setting_smoothing)
                        co2_baseline_more = compute_baseline(co2_list_more, setting_window_size, setting_smoothing)
                        no_baseline_more = compute_baseline(no_list_more, setting_window_size, setting_smoothing)

                        # Override second half of baseline lists with the corresponding value in its corresponding baseline_more list
                        no2_baseline = overwrite_last_half(no2_baseline, no2_baseline_more)
                        ufp_baseline = overwrite_last_half(ufp_baseline, ufp_baseline_more)
                        o3_baseline = overwrite_last_half(o3_baseline, o3_baseline_more)
                        co_baseline = overwrite_last_half(co_baseline, co_baseline_more)
                        co2_baseline = overwrite_last_half(co2_baseline, co2_baseline_more)
                        no_baseline = overwrite_last_half(no_baseline, no_baseline_more)

                    output_csv = output_file(number)
                    with open(output_csv, "a", newline='') as f:
                        w = csv.writer(f)

                        if current_chunk == 0:

                            # Write settings to output
                            if settings_in_output:
                                w.writerow(['window_size: ', setting_window_size])
                                w.writerow(['smoothing_index: ', setting_smoothing])
                                w.writerow(['chunk_size: ', queue_size])
                                w.writerow(['interlace_chunks: ', interlace_chunks])
                                w.writerow(['', ''])

                            w.writerow(output_cols)

                        for i in range(0, len(no2_list)):
                            w.writerow(
                                [latitude_list[i], longitude_list[i], time_list[i], no2_list[i], ufp_list[i],
                                 o3_list[i], co_list[i], co2_list[i],
                                 no_list[i], ws_list[i], wd_list[i], wv_list[i], "", no2_baseline[i], ufp_baseline[i],
                                 o3_baseline[i],
                                 co_baseline[i], co2_baseline[i], no_baseline[i]])

                    # Break loop if we're on the last chunk, otherwise go to next chunk
                    if len(no2_list) < queue_size:
                        print("Baseline chunk " + str(current_chunk + 1) + " written")
                        break
                    else:
                        print("Baseline chunk " + str(current_chunk + 1) + " written")
                        current_chunk += 1
            else:
                while True:
                    # Read in the current chunk
                    data = pd.read_csv(folder_directory + input_filenames[number], names=col_names, skiprows=(1 + current_chunk * queue_size),
                                       nrows=queue_size)

                    # Convert current chunk to lists
                    no2_list = data["NO2 (ppb)"].to_list()
                    ufp_list = data["UFP (#/cm^3)"].to_list()
                    o3_list = data["O3 (ppb)"].to_list()
                    co_list = data["CO (ppm)"].to_list()
                    co2_list = data["CO2 (ppm)"].to_list()
                    no_list = data["NO (ppb)"].to_list()
                    ws_list = data["WS (m/s)"].to_list()
                    wd_list = data["WD (degrees)"].to_list()
                    wv_list = data["WV (m/s)"].to_list()
                    time_list = data["date"].to_list()
                    latitude_list = data["latitude"].to_list()
                    longitude_list = data["longitude"].to_list()

                    # Compute baseline for current chunk and save as its own list
                    no2_baseline = compute_baseline(no2_list, setting_window_size, setting_smoothing)
                    ufp_baseline = compute_baseline(ufp_list, setting_window_size, setting_smoothing)
                    o3_baseline = compute_baseline(o3_list, setting_window_size, setting_smoothing)
                    co_baseline = compute_baseline(co_list, setting_window_size, setting_smoothing)
                    co2_baseline = compute_baseline(co2_list, setting_window_size, setting_smoothing)
                    no_baseline = compute_baseline(no_list, setting_window_size, setting_smoothing)

                    output_csv = output_file(number)
                    with open(output_csv, "a", newline='') as f:
                        w = csv.writer(f)

                        if current_chunk == 0:

                            # write settings to output
                            if settings_in_output:
                                w.writerow(['window_size: ', setting_window_size])
                                w.writerow(['smoothing_index: ', setting_smoothing])
                                w.writerow(['chunk_size: ', queue_size])
                                w.writerow(['interlace_chunks: ', interlace_chunks])
                                w.writerow(['', ''])

                            w.writerow(output_cols)

                        for i in range(0, len(no2_list)):
                            w.writerow(
                                [latitude_list[i], longitude_list[i], time_list[i], no2_list[i], ufp_list[i],
                                 o3_list[i], co_list[i], co2_list[i],
                                 no_list[i], ws_list[i], wd_list[i], wv_list[i], "", no2_baseline[i], ufp_baseline[i],
                                 o3_baseline[i],
                                 co_baseline[i], co2_baseline[i], no_baseline[i]])

                    # Break loop if we're on the last chunk, otherwise go to next chunk
                    if len(no2_list) < queue_size:
                        print("Baseline chunk " + str(current_chunk + 1) + " written")
                        break
                    else:
                        print("Baseline chunk " + str(current_chunk + 1) + " written")
                        current_chunk += 1

print("")
print("Baseline files created. Analyzing...")

# Get Merged AQ + GPS file names and dates
source_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Baselines\\"
file_names = os.listdir(source_folder)

# Baseline Merged AQ + GPS data (file names):
Baseline_merged_names = [file_name for file_name in file_names if re.match(r"Baseline_MERGED_AQ_GPS_", file_name)]
Baseline_merged_names = [os.path.splitext(file_name)[0] for file_name in Baseline_merged_names]  # Get rid of the .csv
print("")
print("Number of files created are: ", len(Baseline_merged_names))
# Dates:
date_pattern = r"\d{4}_\d{2}_\d{2}"  # Pattern to match the date in the format YYYY_MM_DD
date_list = []
for string in Baseline_merged_names:
    match = re.search(date_pattern, string)
    if match:
        date = match.group()  # Extract the matched date
        date_list.append(date)  # Add the date to the list

date_list = list(set(date_list))
print("Dates in those files are: ", date_list)
print("")
baseline_analysis = []

if enable_baseline_comparison:
    print("Baseline comparison enabled. Comparing...")
    print("")
    for date in date_list:
        # Prefix string
        prefix = f"Baseline_MERGED_AQ_GPS_{date}"

        Baseline_date_and_names = [filename for filename in os.listdir(source_folder) if filename.startswith(prefix)]

        pollutants = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
        baseline_pollutants = ["NO2 baseline (ppb)", "UFP baseline (#/cm^3)", "O3 baseline (ppb)", "CO baseline (ppm)",
                               "CO2 baseline (ppm)", "NO baseline (ppb)"]

        # Create a list to store the best combinations information
        best_combinations = []

        # Create an ExcelWriter object to save the dataframes to an Excel file
        writer = pd.ExcelWriter(source_folder + f'PSNR_analysis_{date}.xlsx', engine='xlsxwriter')
        writer_2 = pd.ExcelWriter(source_folder + f'PSNR_analysis_files_{date}.xlsx', engine='xlsxwriter')

        # Iterate over each pollutant column
        for pollutant_num in range(len(pollutants)):
            best_psnr = 99999.0
            best_file = None
            pollutant_in_action = pollutants[pollutant_num]
            baseline_in_action = baseline_pollutants[pollutant_num]

            # Start a (pollutants, PSNR, Smoothing Index) file lists
            pol = []
            pol_psnr = []
            pol_si = []
            file_list = []

            # Iterate over each CSV file
            for file in Baseline_date_and_names:
                # Read the CSV file
                data = pd.read_csv(source_folder + file, low_memory=False)
                columns_to_keep = ["latitude", "longitude", "date", "NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)",
                                   "CO2 (ppm)", "NO (ppb)", "NO2 baseline (ppb)",
                                   "UFP baseline (#/cm^3)", "O3 baseline (ppb)", "CO baseline (ppm)",
                                   "CO2 baseline (ppm)", "NO baseline (ppb)"]
                data = data[columns_to_keep]

                # Extract true values and baseline predictions for the current pollutant
                true_values = data[pollutant_in_action]
                baseline_predictions = data[baseline_in_action]

                # Get the maximum value from the true values column
                data_range = np.amax(true_values)

                # Calculate the PSNR value for the current pollutant and CSV file
                psnr = peak_signal_noise_ratio(true_values, baseline_predictions, data_range=data_range)
                start_index = file.find("window_size")
                end_index = file.find(".csv")
                middle_part = file[start_index:end_index]
                print(f"For pollutant {pollutant_in_action}, PSNR is {psnr} for settings {middle_part}")

                # Get the SI
                start_index = file.find("smoothing_index = ")
                end_index = file.find(", chunk_size=")
                si = file[start_index:end_index]
                si = int(re.sub(r'\D', '', si))  # Remove string and characters, keep only numbers, convert to integer

                # Append the psnr to pol_psnr list etc.
                pol.append(pollutant_in_action)
                pol_psnr.append(psnr)
                pol_si.append(si)
                file_list.append(file)

            # Create two aux dataframes using the lists (SI = smoothing indexes)
            aux_df = pd.DataFrame({'Pollutant': pol, 'SI': pol_si, 'PSNR': pol_psnr})
            aux_files_df = pd.DataFrame({'Pollutant': pol, 'SI': pol_si, 'file': file_list})

            # Sort the dataframe based on 'SI' in ascending order
            aux_df_sorted = aux_df.sort_values('SI')
            aux_files_df_sorted = aux_files_df.sort_values('SI')

            si = aux_df_sorted['SI'].reset_index(drop=True)
            psnr = aux_df_sorted['PSNR'].reset_index(drop=True)
            files = aux_files_df_sorted['file'].reset_index(drop=True)

            # Calculate the derivative using numpy
            derivative = np.gradient(psnr, si)

            aux_df_sorted['Derivative'] = derivative

            # Find the SI values index where derivative = -0.5
            derivative_list = np.array(derivative).tolist()
            closest_index = find_closest_index(derivative_list)

            # Write the dataframe to a sheet in the Excel file
            aux_df_sorted.to_excel(writer, sheet_name=pollutant_in_action[:3], index=False)
            aux_files_df_sorted.to_excel(writer_2, sheet_name=pollutant_in_action[:3], index=False)

            # Getting index
            print(f"For pollutant {pollutant_in_action} and date {date}, the closest-index is: {closest_index}")
            matching_si_values = si[closest_index]

            best_psnr = psnr[closest_index]
            best_file = files[closest_index]

            # Print the best baseline prediction for the current pollutant
            print(f"For pollutant {pollutant_in_action}, .csv file {best_file} has the best baseline prediction.")
            baseline_analysis.append(
                f"For {date} and pollutant {pollutant_in_action}, .csv file {best_file} has the best baseline prediction with PSNR of {best_psnr}.")

            # Append the best combination of pollutant and baseline to the list
            best_combinations.append({"pollutant": pollutant_in_action, "baseline_file": best_file, "psnr": best_psnr})

        # Save the PSNR_analysis.xslx Excel
        writer.close()

        # Save the best baseline predictions to a text file
        filename = "Analysis_of_Best_Baseline_Predictions.txt"
        full_filename = source_folder + filename
        with open(full_filename, "w") as base_file:
            for prediction in baseline_analysis:
                base_file.write(prediction + "\n")

        # Create a dataframe from the list of best combinations
        df_best_combinations = pd.DataFrame(best_combinations)

        # Create a new dataframe with consecutive columns of pollutant time series and baseline time series
        df_output = pd.DataFrame()

        # Add the date, latitude, and longitude columns to the output dataframe
        df_output["date"] = data["date"]
        df_output["latitude"] = data["latitude"]
        df_output["longitude"] = data["longitude"]

        # Read the .csv with the summary of baseline analysis to create a best-baseline .csv
        for row in df_best_combinations.itertuples():
            pollutant_col = row.pollutant
            baseline_file = row.baseline_file

            # Read the CSV file with the best baseline for the current pollutant
            best_baseline_data = pd.read_csv(source_folder + baseline_file)
            baseline_col = baseline_pollutants[pollutants.index(pollutant_col)]
            baseline_values = best_baseline_data[baseline_col]

            # Add the pollutant and baseline columns to the output dataframe
            df_output[pollutant_col] = data[pollutant_col]
            df_output[baseline_col] = baseline_values

        # Save the best baseline predictions and their details to a CSV file for the current date
        print("")
        print("Baseline comparison concluded. New files saved.")
        filename = f"Best_Baseline_Predictions_{date}.csv"
        full_filename = source_folder + filename
        df_output.to_csv(full_filename, index=False)
else:
    print("Baseline comparison not enabled")

if enable_baseline_subtraction:
    print("")
    print("Baseline subtraction enabled")
    print("")
    for date in date_list:
        file = f"Best_Baseline_Predictions_{date}.csv"
        data = pd.read_csv(source_folder + file)
        pollutants = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
        baseline_pollutants = ["NO2 baseline (ppb)", "UFP baseline (#/cm^3)", "O3 baseline (ppb)", "CO baseline (ppm)",
                               "CO2 baseline (ppm)", "NO baseline (ppb)"]
        pollutants_signal = ["NO2 signal (ppb)", "UFP signal (#/cm^3)", "O3 signal (ppb)", "CO signal (ppm)",
                             "CO2 signal (ppm)", "NO signal (ppb)"]

        for pollutant_num in range(len(pollutants)):
            print(f"Subtracting baseline for {pollutants[pollutant_num]}")
            data[pollutants_signal[pollutant_num]] = data[pollutants[pollutant_num]] - data[
                baseline_pollutants[pollutant_num]]

        columns_to_keep = ["latitude", "longitude", "date", "NO2 signal (ppb)", "UFP signal (#/cm^3)",
                           "O3 signal (ppb)", "CO signal (ppm)",
                           "CO2 signal (ppm)", "NO signal (ppb)"]

        data = data[columns_to_keep]

        # Save the best baseline predictions and their details to a CSV file for the current date
        print("Baseline subtraction concluded. New files saved.")
        filename = f"Signal_After_Best_Baseline_Subtraction_{date}.csv"
        full_filename = source_folder + filename
        data.to_csv(full_filename, index=False)
else:
    print("Baseline subtraction not enabled")

if plotting_baselines_allfiles:
    print("")
    print("Baseline plotting enabled")
    print("")
    output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Figures\\Baselines\\"

    for date in date_list:
        # Prefix string
        prefix = f"Baseline_MERGED_AQ_GPS_{date}"

        # Get the list of filenames that start with the specified prefix
        matching_files = [filename for filename in os.listdir(source_folder) if filename.startswith(prefix)]

        # Read each matching CSV file into a dataframe
        for filename in matching_files:
            file_path = os.path.join(source_folder, filename)
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')

            pollutants = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
            baseline_pollutants = ["NO2 baseline (ppb)", "UFP baseline (#/cm^3)", "O3 baseline (ppb)",
                                   "CO baseline (ppm)",
                                   "CO2 baseline (ppm)", "NO baseline (ppb)"]

            for pollutant_num in range(len(pollutants)):
                print(f"Plotting baseline for {date} and {pollutants[pollutant_num]} in {filename}")
                # Extract the x-axis values from the "date" column
                x = data["date"]
                # Extract the y-axis values from the columns you want to plot
                y1 = data[pollutants[pollutant_num]]
                y2 = data[baseline_pollutants[pollutant_num]]
                # Create the plot
                fig, ax = plt.subplots(figsize=(16, 5))
                ax.plot(x, y1, label=pollutants[pollutant_num], color="black")
                ax.plot(x, y2, label=baseline_pollutants[pollutant_num], color="red")
                # Format the x-axis tick labels
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                # Add a title to the plot
                plt.title(f"Baseline for {pollutants[pollutant_num]} in {date}")
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
                start_index = filename.find("window_size")
                end_index = filename.find(".csv")
                middle_part = filename[start_index:end_index]
                pollutants_newlist = ["NO2 (ppb)", "UFP (counts)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
                plt.savefig(output_folder + pollutants_newlist[pollutant_num] + "_" + date + "_" + middle_part, dpi=500,
                            transparent=False)
                # Close the plot
                plt.close()
else:
    print("Plotting ALL baselines not enabled")

if plotting_baselines_bestfiles:
    print("")
    print("Best baseline plotting enabled")
    print("")
    output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Figures\\Baselines\\"

    for date in date_list:
        # Prefix string
        prefix = f"Best_Baseline_Predictions_{date}"

        # Get the list of filenames that start with the specified prefix
        matching_files = [filename for filename in os.listdir(source_folder) if filename.startswith(prefix)]

        # Read each matching CSV file into a dataframe
        for filename in matching_files:
            file_path = os.path.join(source_folder, filename)
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')

            pollutants = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
            baseline_pollutants = ["NO2 baseline (ppb)", "UFP baseline (#/cm^3)", "O3 baseline (ppb)",
                                   "CO baseline (ppm)",
                                   "CO2 baseline (ppm)", "NO baseline (ppb)"]

            for pollutant_num in range(len(pollutants)):
                print(f"Plotting BEST baseline for {date} and {pollutants[pollutant_num]} in {filename}")
                # Extract the x-axis values from the "date" column
                x = data["date"]
                # Extract the y-axis values from the columns you want to plot
                y1 = data[pollutants[pollutant_num]]
                y2 = data[baseline_pollutants[pollutant_num]]
                # Create the plot
                fig, ax = plt.subplots(figsize=(16, 5))
                ax.plot(x, y1, label=pollutants[pollutant_num], color="black")
                ax.plot(x, y2, label=baseline_pollutants[pollutant_num], color="red")
                # Format the x-axis tick labels
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                # Add a title to the plot
                plt.title(f"Best baseline for for {pollutants[pollutant_num]} in {date}")
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
                pollutants_newlist = ["NO2 (ppb)", "UFP (counts)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
                plt.savefig(output_folder + f"Best_baseline_for_" + date + "_" + pollutants_newlist[pollutant_num],
                            dpi=500, transparent=False)
                # Close the plot
                plt.close()
else:
    print("Plotting BEST baselines not enabled")

if plotting_psnr_analysis:
    for date in date_list:
        # Specify the file path of the Excel spreadsheet
        file_path = f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Baselines\\PSNR_analysis_{date}.xlsx'

        # Read the Excel file into a dictionary of dataframes
        dfs = pd.read_excel(file_path, sheet_name=None)

        # Iterate over each sheet in the dictionary and create a plot
        for sheet_name, df in dfs.items():
            # Extract the necessary columns from the dataframe
            pollutant = df['Pollutant'].iloc[0]
            si = df['SI']
            psnr = df['PSNR']

            # Calculate the derivative using numpy
            derivative = np.gradient(psnr, si)

            # Create the plot
            fig, ax1 = plt.subplots()

            # Plot the primary y-axis (PSNR)
            ax1.plot(si, psnr, 'b')
            ax1.set_xlabel('SI')
            ax1.set_ylabel('PSNR', color='b')

            # Create the secondary y-axis for the derivative
            ax2 = ax1.twinx()

            # Plot the secondary y-axis (Derivative)
            ax2.plot(si, derivative, 'r')
            ax2.set_ylabel('Derivative', color='r')

            # Set the title
            plt.title(f'Plot for {pollutant}')

            # Add minor ticks to the y-axis and x-axis
            ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())

            # Add gridlines
            ax1.grid(which='both', color='lightgray', linestyle='--')
            ax2.grid(which='both', color='lightgray', linestyle='--')

            # Display or save the plot
            plt.savefig(
                f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Figures\\Baselines\\PSNR_analysis_{pollutant[:3]}_{date}",
                dpi=500, transparent=False)

            # Close the plot
            plt.close()
else:
    print("")
    print("PNSR analysis not plotted.")

if save_all_baselines:
    print("")
    print("The individual Baseline files created in each run were kept.")
else:
    print("")
    print("The individual Baseline files created in each run were excluded.")
    folder_path = source_folder  # Specify the folder path where the CSV files are located

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over each file in the folder
    for file_name in file_list:
        if file_name.endswith('.csv') and file_name.startswith('Baseline_MERGED_AQ_GPS_'):
            file_path = os.path.join(folder_path, file_name)  # Get the full file path
            os.remove(file_path)  # Delete the file


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

