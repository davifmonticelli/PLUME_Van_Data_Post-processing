# This script was created to automatically post-process P.L.U.M.E. Van data after multiple days of sampling
# Function: Uses the merged (Sensor transcript + GPS) to markdown plumes in the timeseries
# Authors: Chris Kelly and Davi de Ferreyro Monticelli, iREACH group (University of British Columbia)
# Date: 2023-07-06
# Version: 1.0.0

# THIS IS A NEW FEATURE:
# A PLUME IS DEFINED AS A STEADY (SLOW) INCREASE FOLLOWED BY A STEADY (SLOW) DECREASE IN CONCENTRATION.

# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "Pre-processing_PLUME_Data.py" PRIOR
# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "auto_merge_data.py" PRIOR

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import re
import sys
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from scipy.signal import find_peaks, peak_prominences
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
file_path_txt = f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Plumes\\Plume_console_output.txt'

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

# Get Merged AQ + GPS file names and dates
source_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\"
output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Plumes\\"
file_names = os.listdir(source_folder)

# Merged AQ + GPS data (file names):
Merged_names = [file_name for file_name in file_names if re.match(r"MERGED_AQ_GPS_\d{4}_\d{2}_\d{2}.csv", file_name)]
Merged_names = [os.path.splitext(file_name)[0] for file_name in Merged_names]  # Get rid of the .csv
print("Files processed are: ", Merged_names)
# Dates:
Merged_dates = list(set([re.sub(r"MERGED_AQ_GPS_", "", var_name) for var_name in Merged_names]))
print("Dates processed are: ", Merged_dates)

# Define the threshold (in seconds) values to test
# Used to create a trend line (rolling average)
# Used to evaluate the # points before and after
threshold_values = [600]  # 600 sec. (can be more than one value, e.g. [10, 20, 30])

# Define plume cut points to test
cut_points = [0.5, 0.9]  # 50% and 90% of datapoints before and after (can be more than one value e.g. [0.5, 0.9])

# Define plateau (near flat line at the plume max) size in seconds
plateau_size = 60  # 1 min, applicable for mobile monitoring

# Save plots of Plume Detection?
save_plots = True

########################################################################################################################
# Declaring functions (helpers) for the script
########################################################################################################################

# Define a function to detect plumes
def detect_plumes(data, pollutant_column, threshold, cut_point, plateau_size):

    # 1st Decompose timeseries
    decomposition = seasonal_decompose(data[pollutant_column], model='additive', period=threshold)
    trend = decomposition.trend
    trend_modified = trend.fillna(0)  # Replace NaN values with 0 because find_peaks() does not work well with NaN

    # Find peaks in the trend line -- also considering a plateau from 1*threshold to 60*threshold:
    peaks, _ = find_peaks(trend_modified.values, plateau_size=(1, plateau_size))

    # Initialize a list to store the plume intervals
    plume_intervals = []

    #### Check for plume intervals ####
    # Iterate over the time series data

    '''
    # If desired a longer operation that look at every datapoint
    concentrations = data[pollutant_column].values  # Needed if commented section below is used
    for i in range(threshold, len(concentrations)-threshold):
        # Establish count based on the cut_point informed:
        diff_minus = trend[i - threshold:i] < trend[i]
        diff_plus = trend[i] < trend[i + 1:i + threshold + 1]
        count_minus = np.sum(diff_minus)
        count_plus = np.sum(diff_plus)
    '''

    # For every peak found, it will evaluate if it is a peak or a plume and append the true plume intervals
    for i in peaks:
        # Get the value at the trend-peak index
        value = trend.iloc[i]
        # Count the number of values after the arbitrary index that are lower than the value at the arbitrary index
        count_plus = len(trend[(trend < value) & (trend.index > i)].iloc[:threshold])
        # Count the number of values before the arbitrary index that are lower than the value at the arbitrary index
        count_minus = len(trend[(trend < value) & (trend.index < i)].iloc[-threshold:])

        # Check if the trend increases within the time threshold and then decreases or remains stationary
        # If (cut_point*100%) of the values before the point trend[i] indicate increase,
        # and the (cut_point*100%) after indicate decrease, then:
        if (count_minus >= (cut_point * threshold)) and (count_plus >= (cut_point * threshold)):
            plume_intervals.append(data.iloc[i - threshold]['date'])
            plume_intervals.append(data.iloc[i + threshold]['date'])
            set(list(plume_intervals))

    return plume_intervals, trend

# Create function to create a new file to save the plume intervals
def output_txt(date_to_run):
    date = date_to_run
    output_file = output_folder+f"Plume_Intervals_for_{date}.txt"
    return output_file

def read_Merged_files(date_to_run):
    # Read MERGED_AQ_GPS_XXXX_XX_XX csv files
    date = date_to_run
    file_path = source_folder
    file_to_read = f"MERGED_AQ_GPS_{date}.csv"
    sensor_data = pd.read_csv(file_path+file_to_read)

    return sensor_data

########################################################################################################################
# Main script
########################################################################################################################

# Create a dictionary to store the dataframes per threshold value
plume_dataframes = {}

# Iterate over the dates and threshold values
for date in Merged_dates:
    output_file = output_txt(date)
    data = read_Merged_files(date)
    #data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
    print("")
    print(f"Processing dataframe: MERGED_AQ_GPS_{date}.csv")
    print(data)

    with open(output_file, "w") as file:
        for cut_point in cut_points:

            for threshold in threshold_values:
                file.write(f"For plume threshold = {threshold} seconds, and cut point = {cut_point}, the plumes are found at:\n")

                # Create a new dataframe for the current threshold value
                new_dataframe = data.copy()
                new_dataframe[""] = ""

                # Iterate over the pollutant columns
                pollutant_columns = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]

                print("Calculating trends and plumes for ", date, "timeseries")
                for pollutant_column in pollutant_columns:
                    print("Inspecting plume intervals for pollutant: ", pollutant_column, ", for time threshold: ",
                          threshold, "sec. and cut point: ", cut_point * 100, "%")
                    plume_intervals, plume_trend = detect_plumes(data, pollutant_column, threshold, cut_point, plateau_size)

                    if len(plume_intervals) > 0:
                        selected_data = []
                        print("Arranging/sorting plumes...")
                        # Get pollutant values within intervals of plume (several, creates duplicates)
                        formatted_intervals = [datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S') for date_str in plume_intervals]

                        print(f"Relevant intervals arranged/sorted. There are {int(len(formatted_intervals)/2)} plumes in this dataset.")

                        data['date'] = pd.to_datetime(data['date'])

                        for k in range(0, len(formatted_intervals), 2):
                            # Get pollutant values within intervals of plume (several, creates duplicates)
                            mask = data['date'].between(formatted_intervals[k], formatted_intervals[k + 1])
                            selected_data.append(data.loc[mask, pollutant_column])

                        print("Creating new dataframe...")
                        print("")
                        # Concatenate all selected data into a single DataFrame
                        selected_data = pd.concat(selected_data, ignore_index=False)
                        # Keep only the first occurrence of each index
                        selected_data = selected_data.groupby(selected_data.index).first()
                        # Total number of rows in the new DataFrame
                        total_rows = len(data)
                        # Reindex the DataFrame with the desired total number of rows and fill the blanks with NaN
                        plume_df_filled = selected_data.reindex(range(0, total_rows), fill_value=np.nan)

                    # Update the new dataframe with plume information
                    new_dataframe[f'{pollutant_column}_plume'] = plume_df_filled

                    # Add pollutant trend to dataframe
                    new_dataframe[f'{pollutant_column}_trend'] = plume_trend

                    # Write the plume intervals for the current pollutant and threshold to the file
                    file.write(f"Plume intervals for {pollutant_column}:\n")
                    if len(plume_intervals) > 0:
                        file.write(f"{plume_intervals}\n")
                        file.write(f"{formatted_intervals}\n")
                    else:
                        file.write(f"No plume intervals found for {pollutant_column}.\n")
                    file.write("\n")

                file.write("\n")
                plume_dataframes[threshold] = new_dataframe
                plume_dataframes[threshold].to_csv(
                    output_folder + f'Plumes_in_MERGED_AQ_GPS_{date}_for_threshold_{threshold}_sec_and_cut_point_{cut_point}.csv', index=False)

print("Plumes processed and new files created. Analyzing...")

# Get Merged AQ + GPS file names and dates
source_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Plumes\\"
file_names = os.listdir(source_folder)

# Plumes Merged AQ + GPS data (file names):
Plumes_merged_names = [file_name for file_name in file_names if re.match(r"Plumes_in_MERGED_AQ_GPS_", file_name)]
Plumes_merged_names = [os.path.splitext(file_name)[0] for file_name in Plumes_merged_names]  # Get rid of the .csv
print("")
# print("Files created are: ", Plumes_merged_names)
# Dates:
date_pattern = r"\d{4}_\d{2}_\d{2}"  # Pattern to match the date in the format YYYY_MM_DD
date_list = []
for string in Plumes_merged_names:
    match = re.search(date_pattern, string)
    if match:
        date = match.group()  # Extract the matched date
        date_list.append(date)  # Add the date to the list

# Set date list to avoid plotting the same figure multiple times
date_list = [*set(date_list)]
print("Dates processed are: ", date_list)

if save_plots:
    print("")
    print("Plotting timeseries from plume detection algorithm enabled")
    print("")
    output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Figures\\Plumes\\"

    for date in date_list:
        # Prefix string
        prefix = f"Plumes_in_MERGED_AQ_GPS_{date}"

        # Get the list of filenames that start with the specified prefix
        matching_files = [filename for filename in os.listdir(source_folder) if filename.startswith(prefix)]

        # Read each matching CSV file into a dataframe
        for filename in matching_files:
            file_path = os.path.join(source_folder, filename)
            data = pd.read_csv(file_path, low_memory=False)
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')

            pollutants = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
            pollutants_plume = ["NO2 (ppb)_plume", "UFP (#/cm^3)_plume", "O3 (ppb)_plume",
                                "CO (ppm)_plume", "CO2 (ppm)_plume", "NO (ppb)_plume"]
            pollutants_trend = ["NO2 (ppb)_trend", "UFP (#/cm^3)_trend", "O3 (ppb)_trend",
                                "CO (ppm)_trend", "CO2 (ppm)_trend", "NO (ppb)_trend"]

            for pollutant_num in range(len(pollutants)):

                # Plotting pollutant signal + plumes timeseries:
                print(f"Plotting plumes for {date} and {pollutants[pollutant_num]} in {filename}")
                # Extract the x-axis values from the "date" column
                x = data["date"]
                # Extract the y-axis values from the columns you want to plot
                y1 = data[pollutants[pollutant_num]]
                y2 = data[pollutants_plume[pollutant_num]]
                # Create the plot
                fig, ax = plt.subplots(figsize=(16, 5))
                ax.plot(x, y1, label=pollutants[pollutant_num], color="black")
                ax.plot(x, y2, label=pollutants_plume[pollutant_num], color="red")
                # Format the x-axis tick labels
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                # Add a title to the plot
                plt.title(f"Plumes for {pollutants[pollutant_num]} in {date}")
                # Extract threshold value
                start_index = filename.find("for_threshold_") + len("for_threshold_")
                end_index = filename.find("_sec_and_cut")
                threshold = filename[start_index:end_index]
                # Extract cut point value
                start_index = filename.find("cut_point_") + len("cut_point_")
                end_index = filename.find(".csv")
                cut_point = str((float(filename[start_index:end_index])*100))
                # Add suptitle
                plt.suptitle(f"Configuration used is Threshold (sec.): {threshold} and Cut point (%): {cut_point}", fontsize=10, color="gray")
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
                start_index = filename.find("for_threshold_")
                end_index = filename.find(".csv")
                middle_part = filename[start_index:end_index]
                pollutants_newlist = ["NO2 (ppb)", "UFP (counts)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
                #plt.show()
                plt.savefig(output_folder + "Plumes_for_" + pollutants_newlist[pollutant_num] + "_" + date + "_" + middle_part + ".png", dpi=500,
                            transparent=False)
                # Close the plot
                plt.close()

                # Plotting pollutant signal + trend timeseries:
                print(f"Plotting trends for {date} and {pollutants[pollutant_num]} in {filename}")
                # Extract the x-axis values from the "date" column
                x = data["date"]
                # Extract the y-axis values from the columns you want to plot
                y1 = data[pollutants[pollutant_num]]
                y2 = data[pollutants_trend[pollutant_num]]
                # Create the plot
                fig, ax = plt.subplots(figsize=(16, 5))
                ax.plot(x, y1, label=pollutants[pollutant_num], color="black")
                ax.plot(x, y2, label=pollutants_trend[pollutant_num], color="green")
                # Format the x-axis tick labels
                plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                # Add a title to the plot
                plt.title(f"Trend for {pollutants[pollutant_num]} in {date}")
                # Extract threshold value
                start_index = filename.find("for_threshold_") + len("for_threshold_")
                end_index = filename.find("_sec_and_cut")
                threshold = filename[start_index:end_index]
                # Extract cut point value
                start_index = filename.find("cut_point_") + len("cut_point_")
                end_index = filename.find(".csv")
                cut_point = str((float(filename[start_index:end_index])*100))
                # Add suptitle
                plt.suptitle(f"Configuration used is Threshold (sec.): {threshold} and Cut point (%): {cut_point}", fontsize=10, color="gray")
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
                start_index = filename.find("for_threshold_")
                end_index = filename.find(".csv")
                middle_part = filename[start_index:end_index]
                pollutants_newlist = ["NO2 (ppb)", "UFP (counts)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]
                plt.savefig(output_folder + "Trend_for_" + pollutants_newlist[pollutant_num] + "_" + date + "_" + middle_part + ".png", dpi=500,
                            transparent=False)
                # Close the plot
                plt.close()

else:
    print("Saving plume plots not enabled")


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


