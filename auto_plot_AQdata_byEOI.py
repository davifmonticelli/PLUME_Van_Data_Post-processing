# This script was created to automatically post-process P.L.U.M.E. Van data after multiple days of sampling
# Function: Creates numerous figures, of heatmaps to timeseries, highlighting events of interest
# Authors: Chris Kelly and Davi de Ferreyro Monticelli, iREACH group (University of British Columbia)
# Date: 2023-07-26
# Version: 1.0.0

# THIS IS A NEW FEATURE

# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "Pre-processing_PLUME_Data.py" PRIOR
# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "auto_merge_data.py" PRIOR

import re
import io
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mc  # For the legend
import difflib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns; sns.set_theme(style='white')
from matplotlib.colors import LogNorm
# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
from random import randint
import time
from datetime import datetime
register_matplotlib_converters()
pio.kaleido.scope.default_format = "png"
colors = []
for i in range(100):
    alpha = '%02X' % int(255 * 0.1)  # Calculate alpha value as 10% transparency (0.1 * 255)
    colors.append('#' + alpha + '%06X' % randint(0, 0xFFFFFF))
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
file_path_txt = f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Events of Interest\\EOI_console_output.txt'

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
fmps_data_source = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Pre-processing\\Files\\"
output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Events of Interest\\"
output_folder_fig = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Figures\\Events of Interest\\"
file_names = os.listdir(source_folder)

# Merged AQ + GPS data (file names):
Merged_names = [file_name for file_name in file_names if re.match(r"MERGED_AQ_GPS_\d{4}_\d{2}_\d{2}.csv", file_name)]
Merged_names = [os.path.splitext(file_name)[0] for file_name in Merged_names]  # Get rid of the .csv
print("Files processed are: ", Merged_names)
# Dates:
Merged_dates = list(set([re.sub(r"MERGED_AQ_GPS_", "", var_name) for var_name in Merged_names]))
print("Dates processed are: ", Merged_dates)

# Boolean variable that indicate if a particular event of interest is to be evaluated or ALL of them
target_EOI = True
target_EOI_strings = ['TIF', 'TOL', 'JAM']

# Period of influence to create plots (in seconds)
# This will split heatmaps within a timewindow (e.g., if 1 min, it will be 1 min before and after EOI)
# Unless other Event of Interest (not the same string) happens before this 1min setting
split_time = 60

########################################################################################################################
# Declaring functions (helpers) used throughout the script
########################################################################################################################

# Merging function to save time
# Joins AQ data and EOI data (but can be used for other purposes)
def join_data(csv_t1, csv_t2):
    t1 = pd.read_csv(csv_t1).fillna(0)  # !!! Careful if you don't need to replace NaN
    t2 = pd.read_csv(csv_t2).fillna(0)  # !!! Careful if you don't need to replace NaN
    
    t1['date'] = pd.to_datetime(t1['date'])
    t2['date'] = pd.to_datetime(t2['date'])
    
    merged_df = pd.merge(t1, t2, on='date', how='left')
    return merged_df

import pandas as pd

def split_FMPSdf_by_loop(dataframe, the_date_fmps, save_csv=True):
    # Find the indices of rows where "EOI" contains "LOOP"
    loop_indices = dataframe[dataframe['EOI'].str.contains('LOOP', case=False, na=False)].index

    # Create a list to store DataFrames for each interval
    interval_dataframes = []

    # Loop through the indices to create intervals
    for i in range(len(loop_indices)-1):
        start_idx = loop_indices[i]
        end_idx = loop_indices[i+1]
        interval_df = dataframe.loc[start_idx:end_idx-1].copy()  # Create a copy of the interval DataFrame
        interval_dataframes.append(interval_df)

    # Save each interval DataFrame as a CSV file if 'save_csv' option is True
    if save_csv:
        for i, interval_df in enumerate(interval_dataframes):
            interval_df.to_csv(output_folder+f'FMPS_LOOP_{the_date_fmps}_{i+1}.csv', index=False)

    return interval_dataframes

def split_AQdf_by_loop(dataframe, the_date_aq, save_csv=True):
    # Find the indices of rows where "EOI" contains "LOOP"
    loop_indices = dataframe[dataframe['EOI'].str.contains('LOOP', case=False, na=False)].index

    # Create a list to store DataFrames for each interval
    interval_dataframes = []

    # Loop through the indices to create intervals
    for i in range(len(loop_indices)-1):
        start_idx = loop_indices[i]
        end_idx = loop_indices[i+1]
        interval_df = dataframe.loc[start_idx:end_idx-1].copy()  # Create a copy of the interval DataFrame
        interval_dataframes.append(interval_df)

    # Save each interval DataFrame as a CSV file if 'save_csv' option is True
    if save_csv:
        for i, interval_df in enumerate(interval_dataframes):
            interval_df.to_csv(output_folder+f'AQ_LOOP_{the_date_aq}_{i+1}.csv', index=False)

    return interval_dataframes

# Fill one minute before and after for any AQ data and EOI data
def fill_data(csv_merged):
    n = 1
    fill_df = csv_merged
    # Replace rows containing variations of "LOOP" with blank rows (LOOPs are handled separately...)
    fill_df['EOI'] = fill_df['EOI'].where(~fill_df['EOI'].str.contains('LOOP', case=False, na=False), np.nan)

    if target_EOI:
        # Create a boolean mask for matching target_EOI_strings exactly
        mask = fill_df['EOI'].isin(target_EOI_strings)

        # Replace rows that do not match any target_EOI_string with blank rows (NaN)
        fill_df['EOI'] = fill_df['EOI'].where(mask, np.nan)

        # Replace rows that are not target_EOI with blank rows
        # fill_df['EOI'] = fill_df['EOI'].where(fill_df['EOI'].str.contains(target_EOI_string, case=False, na=False), np.nan)
    else:
        fill_df['EOI'] = fill_df['EOI']

    idx = fill_df.columns.get_loc("EOI")

    while n < split_time:  # see settings declaration
        for row in range(0, len(fill_df)-n):

            if not pd.isna(fill_df.iloc[row, idx]):  # Check if there is a EOI at that timestamp (idx = column "EOI"), if there is:

                if "Possible" not in fill_df.iloc[row, idx]:  # Check if row was already filled, if not, enter below
                    
                    if not pd.isna(fill_df.iloc[row+n, idx]):  # Check if there is an EOI at that timestamp + n
                        fill_df.iloc[row+n, idx] = fill_df.iloc[row+n, idx]  # If there is, leave it
                    else:
                        fill_df.iloc[row+n, idx] = "Possible influence of " + fill_df.iloc[row, idx]  # If there is not, add statement
                    
                    if not pd.isna(fill_df.iloc[row-n, idx]):  # Check if there is an EOI at that timestamp - n
                        fill_df.iloc[row-n, idx] = fill_df.iloc[row-n, idx]  # If there is, leave it
                    else:
                        fill_df.iloc[row-n, idx] = "Possible influence of " + fill_df.iloc[row, idx]  # If there is not, add statement
                else:
                    fill_df.iloc[row, idx] = fill_df.iloc[row, idx]
        n = n+1

    return fill_df

# Separate DataFrame by EOI and form a list of array
# Output 1: Array of "Possible influence of EOI Xs"
# Output 2: Dates (yyyy-mm-dd HH:MM:SS) when EOI were detected
def separate_data(dataframe_todo):
    
    separate_this = dataframe_todo
    idx = separate_this.columns.get_loc("EOI")
    idx_2 = separate_this.columns.get_loc("date")
    separate_this_onlyEOI = separate_this['EOI'].tolist()
    separate_this_onlyEOI = [*set(separate_this_onlyEOI)]
    separate_this_onlyEOI = [x for x in separate_this_onlyEOI if str(x) != 'nan']

    Possible_EOIs = [x for x in separate_this_onlyEOI if 'Possible' in str(x)]

    Only_EOIs = [x for x in separate_this_onlyEOI if 'Possible' not in str(x)]

    result_arr = []
    date_aux = []
    
    for row in range(0, len(Possible_EOIs)):
        separate_this_group = separate_this.groupby(['EOI'])
        separate_this_group = separate_this_group.get_group(Possible_EOIs[row])
        separated_frame = separate_this_group.to_numpy()
        result_arr.append(separated_frame)
    
    for k in range(0, len(Only_EOIs)):
        for u in range(0, len(separate_this)):
            if separate_this.iloc[u,idx] == Only_EOIs[k]:
                separate_this_date = separate_this.iloc[u, idx_2]
                separate_this_date = pd.to_datetime(separate_this_date)

                separated_date = separate_this_date.to_numpy()
                date_aux.append(separated_date)
                    
    for l in range(0, len(date_aux)):
        date_aux[l] = pd.Timestamp(date_aux[l])

    return result_arr, date_aux

# Get breaks for plotting
def get_brakes(breakframe):
    breakthis = breakframe
    
    date_intervals = []
    date_intervals.append(breakthis.iloc[0, 0])

    for i in range(1, len(breakthis)):
        x = breakthis.iloc[i, 0] - breakthis.iloc[i-1, 0]
        if x.seconds > 5:
            date_intervals.append(breakthis.iloc[i-1, 0])
            date_intervals.append(breakthis.iloc[i, 0])
            
    date_intervals.append(breakthis.iloc[len(breakthis)-1, 0])
    return date_intervals

# ## Seaborn functions: ## #
# Creating dataframes for seaborn plots for FMPS data
def seaborn_adjust(array_sns):
    frame_sns = pd.DataFrame(array_sns, columns=['date','Dp = 6.04','Dp = 6.98','Dp = 8.06','Dp = 9.31','Dp = 10.8','Dp = 12.4','Dp = 14.3','Dp = 16.5','Dp = 19.1','Dp = 22.1','Dp = 25.5','Dp = 29.4','Dp = 34','Dp = 39.2','Dp = 45.3','Dp = 52.3','Dp = 60.4','Dp = 69.8','Dp = 95.2','Dp = 114.1','Dp = 138.9','Dp = 167.6','Dp = 200.8','Dp = 239.1','Dp = 283.3','Dp = 334.4','Dp = 393.3','Dp = 461.5','Dp = 540.0','Dp = 630.8','Dp = 735.8','Dp = 856.8','EOI'])
    frame_sns.columns = ['date','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','Dp','EOI']
    dps = ['Dp = 6.04','Dp = 6.98','Dp = 8.06','Dp = 9.31','Dp = 10.8','Dp = 12.4','Dp = 14.3','Dp = 16.5','Dp = 19.1','Dp = 22.1','Dp = 25.5','Dp = 29.4','Dp = 34','Dp = 39.2','Dp = 45.3','Dp = 52.3','Dp = 60.4','Dp = 69.8','Dp = 95.2','Dp = 114.1','Dp = 138.9','Dp = 167.6','Dp = 200.8','Dp = 239.1','Dp = 283.3','Dp = 334.4','Dp = 393.3','Dp = 461.5','Dp = 540.0','Dp = 630.8','Dp = 735.8','Dp = 856.8']
    frame_sns_2 = frame_sns.iloc[:, 0:2]
    eois_to_insert = frame_sns.iloc[:, 33]  # FMPS data has 32 columns, column 33 will be the EOI
    frame_sns_2.insert(2, 'EOI', eois_to_insert)
               
    for i in range(1,len(dps)):
        dates_to_append = frame_sns.iloc[:, 0]
        dp_to_append = frame_sns.iloc[:, i+1]  # Don't mess with this +1!!!!
        eois_to_append = frame_sns.iloc[:, 33]  # EOI column is 33 for FMPS data
        df_to_append = pd.concat([dates_to_append, dp_to_append, eois_to_append], axis=1, ignore_index=True)
        df_to_append.columns = ['date', 'Dp', 'EOI']
        frame_sns_2 = pd.concat([frame_sns_2, df_to_append], ignore_index=True)
   
    frame_sns_2.columns = ['date', 'Total particle count', 'EOI']
    # Declaring magnitude of repetition
    K = int(len(frame_sns_2)/32)  # 32 = number of diameters (size bins)
    # Using list comprehension
    # Repeat elements K times
    diameters = [6.04, 6.98, 8.06, 9.31, 10.8, 12.4, 14.3, 16.5, 19.1, 22.1, 25.5, 29.4, 34, 39.2, 45.3, 52.3, 60.4, 69.8, 95.2, 114.1, 138.9, 167.6, 200.8, 239.1, 283.3, 334.4, 393.3, 461.5, 540.0, 630.8, 735.8, 856.8]
    diameters_repeated = []
    for i in diameters:
        for ele in range(K):
            diameters_repeated.append(i)
    
    frame_sns_2.insert(2, 'Size bin', diameters_repeated)
 
    return frame_sns_2

# Creating dataframes for seaborn plots for AQ data
def seaborn_adjust_aq(array_sns):
    frame_sns = pd.DataFrame(array_sns, columns=['lat','long','date','NO2','UFP','O3','CO','CO2','NO','WS','WD','WV','EOI'])
    frame_sns.drop(frame_sns.columns[:2], axis=1, inplace=True)  # Drop first 2 column not needed
    frame_sns.columns = ['date','AQ','AQ','AQ','AQ','AQ','AQ','AQ','AQ','AQ','EOI']
    dps = ['NO2','UFP','O3','CO','CO2','NO']
    frame_sns_2 = frame_sns.iloc[:, 0:2]
    eois_to_insert = frame_sns.iloc[:, 10]
    frame_sns_2.insert(2, 'EOI', eois_to_insert)
               
    for i in range(1, len(dps)):
        dates_to_append = frame_sns.iloc[:, 0]
        dp_to_append = frame_sns.iloc[:, i+1]  # Don't mess with this +1!!!!
        eois_to_append = frame_sns.iloc[:, 10]  # EOI column is 10
        df_to_append = pd.concat([dates_to_append, dp_to_append, eois_to_append], axis=1, ignore_index=True)
        df_to_append.columns = ['date', 'AQ', 'EOI']
        frame_sns_2 = pd.concat([frame_sns_2, df_to_append], ignore_index=True)
   
    frame_sns_2.columns = ['date', 'AQ', 'EOI']
    # Declaring magnitude of repetition
    K = int(len(frame_sns_2)/6)  # 6 = Number of pollutants
    # Using list comprehension
    # Repeat elements K times
    AQ_measured = ['NO2', 'UFP', 'O3', 'CO', 'CO2', 'NO']
    AQ_repeated = []
    for i in AQ_measured:
        for ele in range(K):
            AQ_repeated.append(i)
    
    frame_sns_2.insert(2, 'Air pollutant', AQ_repeated)
    NO2_frame = frame_sns_2[frame_sns_2['Air pollutant'] == 'NO2']
    UFP_frame = frame_sns_2[frame_sns_2['Air pollutant'] == 'UFP']
    O3_frame = frame_sns_2[frame_sns_2['Air pollutant'] == 'O3']
    CO_frame = frame_sns_2[frame_sns_2['Air pollutant'] == 'CO']
    CO2_frame = frame_sns_2[frame_sns_2['Air pollutant'] == 'CO2']
    NO_frame = frame_sns_2[frame_sns_2['Air pollutant'] == 'NO']
     
    return NO2_frame, UFP_frame, O3_frame, CO_frame, CO2_frame, NO_frame

# Creating color_scheme (cmap) for seaborn heatmaps
def NonLinCdict(steps, hexcol_array):
    cdict = {'red': (), 'green': (), 'blue': ()}
    for s, hexcol in zip(steps, hexcol_array):
        rgb = mc.hex2color(hexcol)
        cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
        cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
        cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
    return cdict


# Creates the shades to plot figures (EOI intervals)
def create_shades(dictionary):
    frame_to_shade = dictionary
    framing = pd.DataFrame.from_dict(dictionary)
    rows = len(frame_to_shade['date']) - 1
    frame_to_shade_onlyEOIs = framing['EOI'].tolist()  # EOI column
    frame_to_shade_onlyEOIs = [*set(frame_to_shade_onlyEOIs)]
    frame_to_shade_onlyEOIs = [x for x in frame_to_shade_onlyEOIs if str(x) != 'nan']

    result_intervals = []
    EOI = []
    # Get the intervals per EOI
    for k in range(0, len(frame_to_shade_onlyEOIs)):
        for i in range(1, rows + 1):
            if i == rows + 1:
                frame_to_shade['date'][i] = frame_to_shade['date'][
                    len(frame_to_shade['date']) - 1]  # Use the column with date

            elif frame_to_shade['EOI'][i] == frame_to_shade_onlyEOIs[k]:  # Use the column with the EOI
                EOI.append([frame_to_shade_onlyEOIs[k], frame_to_shade['date'][i - 1], frame_to_shade['date'][i]])

    result_intervals = EOI

    return result_intervals


# Get the list of unique EOIs by dataframe
def eoi_list(data):
    framing = data
    frame_to_shade_onlyEOIs = framing['EOI'].tolist()  # EOI column
    frame_to_shade_onlyEOIs = [*set(frame_to_shade_onlyEOIs)]
    frame_to_shade_onlyEOIs = [x for x in frame_to_shade_onlyEOIs if str(x) != 'nan']

    return frame_to_shade_onlyEOIs


# For plotting:
# New colorscale (0 is transparent)
colorscale_toplot=[[0.0, "rgb(255,255,255)"],
                  [0.1111111111111111, "rgb(72,40,120)"],
                  [0.2222222222222222, "rgb(62,73,137)"],
                  [0.3333333333333333, "rgb(49,104,142)"],
                  [0.4444444444444444, "rgb(38,130,142)"],
                  [0.5555555555555556, "rgb(31,158,137)"],
                  [0.6666666666666666, "rgb(53,183,121)"],
                  [0.7777777777777778, "rgb(110,206,88)"],
                  [0.8888888888888888, "rgb(181,222,43)"],
                  [1.0, "rgb(253,231,37)"]]

seaborn_colors = ["#FFFFFF", "#482878", "#3E4989", "#31688E", "#26828E", "#1F9E89", "#35B779", "#6ECE58", "#76DE2B", "#FDE725"]
seaborn_intervals = [0, 0.1111111111111111, 0.22222222222222222, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556, 0.6666666666666666, 0.7777777777777778, 0.8888888888888888, 1.0]
seaborn_dict = NonLinCdict(seaborn_intervals, seaborn_colors)
seaborn_cmap = mc.LinearSegmentedColormap('Seaborn_cmap', seaborn_dict)

########################################################################################################################
# Main script
########################################################################################################################

sampling_dates = Merged_dates

# # FMPS
# ## Preparing data for Analysis:
print("Preparing FMPS data for analysis...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]
    globals()[f"FMPS_{the_date}_EOIs"] = join_data(fmps_data_source+"FMPS_Corrected_{the_date}_UPDATED.csv".format(the_date=the_date),fmps_data_source+"PLUME_EOIMap_All_Days.csv")
    # Extracting the loops:
    print(f"Extracting the LOOPs for the date: {the_date}...")
    intervals_FMPS = split_FMPSdf_by_loop(globals()[f"FMPS_{the_date}_EOIs"], the_date, save_csv=True)
    # Everything else:
    print(f"Filling events for the date: {the_date}...")
    globals()[f"FMPS_{the_date}_EOIs"] = fill_data(globals()[f"FMPS_{the_date}_EOIs"])
    globals()[f"FMPS_{the_date}_EOIs"].to_csv(output_folder+'FMPS_{the_date}_EOIs.csv'.format(the_date=the_date), index=False)

# ## Loading pre-prepared data:
#for g in range(len(sampling_dates)):
#    the_date = sampling_dates[g]
#    globals()[f"FMPS_{the_date}_EOIs"] = pd.read_csv(output_folder+'FMPS_{the_date}_EOIs.csv'.format(the_date=the_date), dtype={'EOI': 'object'}, header=0)

# ## Creating Heatmaps by EOI Event:
# Separating the DataFrame into groups according to the EOI influence (transforms into list of arrays)
# Groups = FMPS outputs under possible influence of a EOI
# Dates = Exact hh:mm:ss the EOI was detected (for plotting)
print("Creating FMPS data groups...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]
    globals()[f"FMPS_{the_date}_groups"], globals()[f"FMPS_{the_date}_dates"] = separate_data(globals()[f"FMPS_{the_date}_EOIs"])

# #### Plotting the UFP Distribution with EOI Detection Time:
# #### Seaborn preparation
print("Preparing FMPS data heatmaps...")
# Might need to ignore warnings for this one, so:
import warnings
warnings.filterwarnings('ignore')
#######

for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]

    # First create heatmap of the full-day for plotting (then each group, subgroup...)
    # Convert dataframe to numpy
    globals()[f"FMPS_{the_date}_EOIs"] = globals()[f"FMPS_{the_date}_EOIs"].to_numpy()
    # Creates a dataframe from the respective array/list
    globals()[f"FMPS_{the_date}_EOIs"] = seaborn_adjust(globals()[f"FMPS_{the_date}_EOIs"])
    # Fix the dates for transformations and plotting
    globals()[f"FMPS_{the_date}_EOIs"]['date'] = pd.to_datetime(globals()[f"FMPS_{the_date}_EOIs"]['date'])

    # Get the brakes/timestamp for EOI events
    brakes = get_brakes(globals()[f"FMPS_{the_date}_EOIs"])
    brakes = [x for x in brakes if str(x) != 'NaT']  # Sometimes they come with an extra, so this cut it out
    brakes = [*set(brakes)]  # Need to exclude repeated values due to seaborn output format
    brakes.sort()  # Corrects the order
    idx = []
    idx.append(0)
    u = 0

    globals()[f"dates_ibtw_{the_date}"] = []
    while u < len(brakes):
        interval = pd.Interval(left=brakes[u], right=brakes[u + 1])
        u = u + 2
        for k in range(0, len(globals()[f"FMPS_{the_date}_dates"])):
            if globals()[f"FMPS_{the_date}_dates"][k] in interval:
                globals()[f"dates_ibtw_{the_date}"].append(globals()[f"FMPS_{the_date}_dates"][k])

    # Re-order dataframe to be according to date
    globals()[f"FMPS_{the_date}_EOIs"].sort_values(by='date', inplace=True, ignore_index=True)
    # Build Heatmaps for the entire run
    globals()[f"FMPS_{the_date}_EOIs"]['date'] = pd.to_datetime(globals()[f"FMPS_{the_date}_EOIs"]['date']).dt.strftime('%H:%M:%S')
    globals()[f"FMPS_{the_date}_Heatvalues"] = globals()[f"FMPS_{the_date}_EOIs"].pivot(index="Size bin", columns="date", values="Total particle count")
    globals()[f"FMPS_{the_date}_Heatvalues"] = globals()[f"FMPS_{the_date}_Heatvalues"][globals()[f"FMPS_{the_date}_Heatvalues"].columns].astype(float)  # or int

    for i in range(len(globals()[f"FMPS_{the_date}_groups"])):  # Breakes the FMPS result in multiple groups, each Group corresponds to a EOI. Obtain variables for heatmaps.
        globals()[f"FMPS_Group_{i}_{the_date}_subgroups"] = []
        globals()[f"Group_{i}_{the_date}_counts"] = 1
        
        print("")
        globals()[f"FMPS_{the_date}_EOI_Group_{i}"] = globals()[f"FMPS_{the_date}_groups"][i]  # Breaking into groups
        # Creates a dataframe from the respective array/list
        globals()[f"FMPS_{the_date}_EOI_Group_{i}"] = seaborn_adjust(globals()[f"FMPS_{the_date}_EOI_Group_{i}"])
        # Fix the dates for transformations and plotting
        globals()[f"FMPS_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"FMPS_{the_date}_EOI_Group_{i}"]['date'])
        # Get the brakes for EOI events of that Group
        brakes = get_brakes(globals()[f"FMPS_{the_date}_EOI_Group_{i}"])
        brakes = [x for x in brakes if str(x) != 'NaT']  # Sometimes they come with an extra, so this cut it out
        brakes = [*set(brakes)]  # Need to exclude repeated values due to seaborn output format
        brakes.sort()  # Corrects the order
        idx = []
        idx.append(0)
        u = 0

        if len(brakes) > 2:  # Check if subgroups are needed (EOI detected more than once)
            # Creates the Heatmap variables and annotations per subgroup
            for j in range(1, len(brakes)):
                globals()[f"FMPS_{the_date}_EOI_Group_{i}"].sort_values(by='date', inplace=True, ignore_index=True)  # Re-order dataframe to be according to date
                idx.append(int(globals()[f"FMPS_{the_date}_EOI_Group_{i}"][globals()[f"FMPS_{the_date}_EOI_Group_{i}"]['date'] == brakes[j]].index[0]))  # Get the brakes corresponding indexes
                globals()[f"FMPS_{the_date}_EOI_Group_{i}_Sub{j}"] = globals()[f"FMPS_{the_date}_EOI_Group_{i}"].iloc[idx[j-1]:idx[j], :]  # Slices data frame according to brakes
                globals()[f"FMPS_Group_{i}_{the_date}_subgroups"].append(j)  # Auxiliary variable for plotting
                globals()[f"Group_{i}_{the_date}_counts"] = globals()[f"Group_{i}_{the_date}_counts"]+1  # Auxiliary variable for plotting
                
                # Get the dates for the annotation and vertical lines
                globals()[f"dates_ibtw_{the_date}_Group_{i}_Sub{j}"] = []
                if (j % 2) == 1:
                    interval = pd.Interval(left=brakes[j-1], right=brakes[j])
                    for k in range(0, len(globals()[f"FMPS_{the_date}_dates"])):
                        if globals()[f"FMPS_{the_date}_dates"][k] in interval:
                            globals()[f"dates_ibtw_{the_date}_Group_{i}_Sub{j}"].append(globals()[f"FMPS_{the_date}_dates"][k])           

                # Build Heatmaps per subgroup
                globals()[f"FMPS_{the_date}_EOI_Group_{i}_Sub{j}"]['date'] = pd.to_datetime(globals()[f"FMPS_{the_date}_EOI_Group_{i}_Sub{j}"]['date']).dt.strftime('%H:%M:%S')
                globals()[f"FMPS_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"FMPS_{the_date}_EOI_Group_{i}_Sub{j}"].pivot(index="Size bin", columns="date", values="Total particle count")
                globals()[f"FMPS_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"FMPS_{the_date}_Heatvalues_Group_{i}_Sub{j}"][globals()[f"FMPS_{the_date}_Heatvalues_Group_{i}_Sub{j}"].columns].astype(float)  # or int

                # What is what
                if (j % 2) == 1:
                    if "Possible" in globals()[f"FMPS_{the_date}_EOI_Group_{i}"].iloc[0, 3]:  # EOI column
                        globals()[f"EOI_aux_{the_date}_Group_{i}"] = globals()[f"FMPS_{the_date}_EOI_Group_{i}"].iloc[0, 3]
                        print("FMPS_{the_date}_Heatmap_Group_{i}_Sub{j}".format(the_date=the_date, i=i, j=j), " is related to {EOI}".format(EOI=globals()[f"EOI_aux_{the_date}_Group_{i}"]))

        else:  # Rinse and repeat (but without subgroups)
            globals()[f"dates_ibtw_{the_date}_Group_{i}"] = []
            while u < len(brakes):
                interval = pd.Interval(left=brakes[u], right=brakes[u+1])
                u = u+2
                for k in range(0, len(globals()[f"FMPS_{the_date}_dates"])):
                    if globals()[f"FMPS_{the_date}_dates"][k] in interval:
                        globals()[f"dates_ibtw_{the_date}_Group_{i}"].append(globals()[f"FMPS_{the_date}_dates"][k])

            globals()[f"FMPS_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"FMPS_{the_date}_EOI_Group_{i}"]['date']).dt.strftime('%H:%M:%S')
            globals()[f"FMPS_{the_date}_Heatvalues_Group_{i}"] = globals()[f"FMPS_{the_date}_EOI_Group_{i}"].pivot(index="Size bin", columns="date", values="Total particle count")
            globals()[f"FMPS_{the_date}_Heatvalues_Group_{i}"] = globals()[f"FMPS_{the_date}_Heatvalues_Group_{i}"][globals()[f"FMPS_{the_date}_Heatvalues_Group_{i}"].columns].astype(float)  # or int

            # What is what
            if "Possible" in globals()[f"FMPS_{the_date}_EOI_Group_{i}"].iloc[0, 3]:  # EOI column
                globals()[f"EOI_aux_{the_date}_Group_{i}"] = globals()[f"FMPS_{the_date}_EOI_Group_{i}"].iloc[0, 3]
                print("FMPS_{the_date}_Heatmap_Group_{i}".format(the_date=the_date, i=i), " is related to {EOI}".format(EOI=globals()[f"EOI_aux_{the_date}_Group_{i}"]))


# Plotting FMPS groups and subgroups (Seaborn method)
# NOT-Logarithmic scale
print("")
print("Plotting FMPS data in normal scale...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]

    # First plotting the entire run, then each group, subgroups...
    plt.figure(figsize=(16, 5))
    ax = sns.heatmap(globals()[f"FMPS_{the_date}_Heatvalues"], vmin=1, vmax=10000, cmap=seaborn_cmap)
    ax.invert_yaxis()
    ax.text(x=0.5, y=1.1,
            s='Particles Size-Distribution (nm) during entire sampling and multiple EOI',
            fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=1.05, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center',
            va='bottom', transform=ax.transAxes)
    plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Total particles (#/cm\u00b3)', labelpad=-70)

    #for v in range(0, len(
    #        globals()[f"dates_ibtw_{the_date}"])):  # Trick to include vertical lines and annotations
    #    x = globals()[f"dates_ibtw_{the_date}"][v] - pd.Timedelta(1, unit='S')
    #    t = pd.to_datetime(x).strftime('%H:%M:%S')
    #    test = difflib.get_close_matches(t, globals()[f"FMPS_{the_date}_Heatvalues"].columns, n=1, cutoff=0.5)
    #    idx = globals()[f"FMPS_{the_date}_Heatvalues"].columns.get_loc(test[0])
    #    ax.axvline(idx, linewidth=0.5, color='r')

    plt.savefig(output_folder_fig + 'Seaborn_FMPS_Out_{the_date}_Complete_Run.png'.format(the_date=the_date),
                bbox_inches="tight", dpi=500, transparent=False)
    plt.close()

    for h in range(len(globals()[f"FMPS_{the_date}_groups"])):
    
        if globals()[f"Group_{h}_{the_date}_counts"] == 1:

            plt.figure(figsize=(16,5))
            ax = sns.heatmap(globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}"], vmin=1, vmax=10000, cmap=seaborn_cmap)
            ax.invert_yaxis()
            ax.text(x=0.5, y=1.1, s='Particles Size-Distribution (nm) under ' + str(globals()[f"EOI_aux_{the_date}_Group_{h}"]) + " event", fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
            ax.text(x=0.5, y=1.05, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
            plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)
            cbar = ax.collections[0].colorbar
            cbar.set_label('Total particles (#/cm\u00b3)', labelpad=-70)
            for v in range(0, len(globals()[f"dates_ibtw_{the_date}_Group_{h}"])):  # Trick to include vertical lines and annotations
                x = globals()[f"dates_ibtw_{the_date}_Group_{h}"][v]-pd.Timedelta(1, unit='S')
                t = pd.to_datetime(x).strftime('%H:%M:%S')
                test = difflib.get_close_matches(t, globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}"].columns, n=1, cutoff=0.5)
                idx = globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}"].columns.get_loc(test[0])
                ax.axvline(idx, linewidth=2, color='r')

            eoi_fig = str(globals()[f"EOI_aux_{the_date}_Group_{h}"])
            plt.savefig(output_folder_fig+'Seaborn_FMPS_Out_{the_date}_Group_{h}_{eoi_fig}.png'.format(the_date=the_date, h=h, eoi_fig=eoi_fig), bbox_inches="tight", dpi=500, transparent=False)
            plt.close()

        else:

            for i in range(globals()[f"Group_{h}_{the_date}_counts"]):
                if (i % 2) == 1:
                    plt.figure(figsize=(16, 5))
                    ax = sns.heatmap(globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}_Sub{i}"], vmin=1, vmax=10000, cmap=seaborn_cmap)
                    ax.invert_yaxis()
                    ax.text(x=0.5, y=1.1, s='Particles Size-Distribution (nm) under ' + str(globals()[f"EOI_aux_{the_date}_Group_{h}"]) + " event", fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
                    ax.text(x=0.5, y=1.05, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
                    plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)
                    cbar = ax.collections[0].colorbar
                    cbar.set_label('Total particles (#/cm\u00b3)', labelpad=-70)
                    for v in range(0,len(globals()[f"dates_ibtw_{the_date}_Group_{h}_Sub{i}"])):  # Trick to include vertical lines and annotations
                        x = globals()[f"dates_ibtw_{the_date}_Group_{h}_Sub{i}"][v]-pd.Timedelta(1, unit='S')
                        t = pd.to_datetime(x).strftime('%H:%M:%S')
                        test = difflib.get_close_matches(t, globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}_Sub{i}"].columns, n=1, cutoff=0.5)
                        idx = globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}_Sub{i}"].columns.get_loc(test[0])
                        ax.axvline(idx, linewidth=2, color='r')

                    eoi_fig = str(globals()[f"EOI_aux_{the_date}_Group_{h}"])
                    plt.savefig(output_folder_fig+'Seaborn_FMPS_Out_{the_date}_Group_{h}_Sub{i}_{eoi_fig}.png'.format(the_date=the_date, h=h, i=i, eoi_fig=eoi_fig), bbox_inches="tight", dpi=500, transparent=False)
                    plt.close()

# Plotting FMPS groups and subgroups (Seaborn method)
# Logarithmic scale
print("Plotting FMPS data in log scale...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]

    # First plotting the entire run, then each group, subgroups...
    plt.figure(figsize=(16, 5))
    ax = sns.heatmap(globals()[f"FMPS_{the_date}_Heatvalues"], norm=LogNorm(vmin=1, vmax=100000), cmap=seaborn_cmap)
    ax.invert_yaxis()
    ax.text(x=0.5, y=1.1,
            s='Particles Size-Distribution (nm) during entire sampling and multiple EOI',
            fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=1.05, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center',
            va='bottom', transform=ax.transAxes)
    plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Total particles (#/cm\u00b3)', labelpad=-70)

    #for v in range(0, len(
    #        globals()[f"dates_ibtw_{the_date}"])):  # Trick to include vertical lines and annotations
    #    x = globals()[f"dates_ibtw_{the_date}"][v] - pd.Timedelta(1, unit='S')
    #    t = pd.to_datetime(x).strftime('%H:%M:%S')
    #    test = difflib.get_close_matches(t, globals()[f"FMPS_{the_date}_Heatvalues"].columns, n=1, cutoff=0.5)
    #    idx = globals()[f"FMPS_{the_date}_Heatvalues"].columns.get_loc(test[0])
    #    ax.axvline(idx, linewidth=0.5, color='r')

    plt.savefig(output_folder_fig + 'Seaborn_LogNorm_FMPS_Out_{the_date}_Complete_Run.png'.format(the_date=the_date),
                bbox_inches="tight", dpi=500, transparent=False)
    plt.close()

    for h in range(len(globals()[f"FMPS_{the_date}_groups"])):
    
        if globals()[f"Group_{h}_{the_date}_counts"] == 1:

            plt.figure(figsize=(16, 5))
            ax = sns.heatmap(globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}"], norm=LogNorm(vmin=1, vmax=100000), cmap=seaborn_cmap)
            ax.invert_yaxis()
            ax.text(x=0.5, y=1.1, s='Particles Size-Distribution (nm) under ' + str(globals()[f"EOI_aux_{the_date}_Group_{h}"]) + " event", fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
            ax.text(x=0.5, y=1.05, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
            plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)
            cbar = ax.collections[0].colorbar
            cbar.set_label('Total particles (#/cm\u00b3)', labelpad=-70)
            for v in range(0,len(globals()[f"dates_ibtw_{the_date}_Group_{h}"])):  # Trick to include vertical lines and annotations
                x = globals()[f"dates_ibtw_{the_date}_Group_{h}"][v]-pd.Timedelta(1, unit='S')
                t = pd.to_datetime(x).strftime('%H:%M:%S')
                test = difflib.get_close_matches(t, globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}"].columns, n=1, cutoff=0.5)
                idx = globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}"].columns.get_loc(test[0])
                ax.axvline(idx, linewidth=2, color='r')

            eoi_fig = str(globals()[f"EOI_aux_{the_date}_Group_{h}"])
            plt.savefig(output_folder_fig+'Seaborn_LogNorm_FMPS_Out_{the_date}_Group_{h}_{eoi_fig}.png'.format(the_date=the_date, h=h, eoi_fig=eoi_fig), bbox_inches="tight", dpi=500, transparent=False)
            plt.close()

        else:

            for i in range(globals()[f"Group_{h}_{the_date}_counts"]):
                if (i % 2) == 1:
                    plt.figure(figsize=(16, 5))
                    ax = sns.heatmap(globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}_Sub{i}"], norm=LogNorm(vmin=1, vmax=100000), cmap=seaborn_cmap)
                    ax.invert_yaxis()
                    ax.text(x=0.5, y=1.1, s='Particles Size-Distribution (nm) under ' + str(globals()[f"EOI_aux_{the_date}_Group_{h}"]) + " event", fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
                    ax.text(x=0.5, y=1.05, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
                    plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)
                    cbar = ax.collections[0].colorbar
                    cbar.set_label('Total particles (#/cm\u00b3)', labelpad=-70)
                    for v in range(0, len(globals()[f"dates_ibtw_{the_date}_Group_{h}_Sub{i}"])):  # Trick to include vertical lines and annotations...
                        x = globals()[f"dates_ibtw_{the_date}_Group_{h}_Sub{i}"][v]-pd.Timedelta(1, unit='S')
                        t = pd.to_datetime(x).strftime('%H:%M:%S')
                        test = difflib.get_close_matches(t, globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}_Sub{i}"].columns, n=1, cutoff=0.5)
                        idx = globals()[f"FMPS_{the_date}_Heatvalues_Group_{h}_Sub{i}"].columns.get_loc(test[0])
                        ax.axvline(idx, linewidth=2, color='r')

                    eoi_fig = str(globals()[f"EOI_aux_{the_date}_Group_{h}"])
                    plt.savefig(output_folder_fig+'Seaborn_LogNorm_FMPS_Out_{the_date}_Group_{h}_Sub{i}_{eoi_fig}.png'.format(the_date=the_date, h=h, i=i, eoi_fig=eoi_fig), bbox_inches="tight", dpi=500, transparent=False)
                    plt.close()


# Other AQ Indicators + EOI
# ## Pre-process data:
print("Preparing AQ data for analysis...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]
    globals()[f"AQ_{the_date}_EOIs"] = join_data(source_folder+"Merged_AQ_GPS_{the_date}.csv".format(the_date=the_date),fmps_data_source+"PLUME_EOIMap_All_Days.csv")
    # Extracting the loops:
    print(f"Extracting the LOOPs for date {the_date}...")
    intervals_FMPS = split_AQdf_by_loop(globals()[f"AQ_{the_date}_EOIs"], the_date, save_csv=True)
    # Everything else:
    print(f"Filling events for date {the_date}...")
    globals()[f"AQ_{the_date}_EOIs"] = fill_data(globals()[f"AQ_{the_date}_EOIs"])
    globals()[f"AQ_{the_date}_EOIs"].to_csv(output_folder+'AQ_{the_date}_EOIs.csv'.format(the_date=the_date), index=False)

# ## Load pre-processed data:
#for g in range(len(sampling_dates)):
#    the_date = sampling_dates[g]
#    globals()[f"AQ_{the_date}_EOIs"] = pd.read_csv(output_folder+'AQ_{the_date}_EOIs.csv'.format(the_date=the_date), dtype={'EOI': 'object'}, header=0)

# ## Creating Timeseries by EOI Event:
# Separating the DataFrame into groups according to the EOI influence (transforms into list of arrays)
# Groups = AQ outputs under possible influence of an EOI
# Dates = Exact hh:mm:ss the EOI was detected (for plotting)
print("")
print("Preparing AQ data groups...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]
    globals()[f"AQ_{the_date}_groups"], globals()[f"AQ_{the_date}_dates"] = separate_data(globals()[f"AQ_{the_date}_EOIs"])


# ### Creating Timeseries plots
# Might need to ignore warnings for this one, so:
import warnings
warnings.filterwarnings('ignore')
#######

print("Preparing AQ data heatmaps...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]
    
    for i in range(len(globals()[f"AQ_{the_date}_groups"])):  # Breaks the AQ result in multiple groups_norm, each Group corresponds to an EOI. Obtain variables for heatmaps.
        globals()[f"Group_{i}_{the_date}_counts"] = 1
        print("")
        
        # Divide and conquer:
        globals()[f"AQ_{the_date}_EOI_Group_{i}"] = globals()[f"AQ_{the_date}_groups"][i]  # Breaking into groups
        
        # Creates a data frame from the respective array/list:
        globals()[f"NO2_{the_date}_EOI_Group_{i}"], globals()[f"UFP_{the_date}_EOI_Group_{i}"], globals()[f"O3_{the_date}_EOI_Group_{i}"], globals()[f"CO_{the_date}_EOI_Group_{i}"], globals()[f"CO2_{the_date}_EOI_Group_{i}"], globals()[f"NO_{the_date}_EOI_Group_{i}"] = seaborn_adjust_aq(globals()[f"AQ_{the_date}_EOI_Group_{i}"])
        
        # Fix the dates for transformations and plotting
        globals()[f"NO2_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"NO2_{the_date}_EOI_Group_{i}"]['date'])
        globals()[f"UFP_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"UFP_{the_date}_EOI_Group_{i}"]['date'])
        globals()[f"O3_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"O3_{the_date}_EOI_Group_{i}"]['date'])
        globals()[f"CO_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"CO_{the_date}_EOI_Group_{i}"]['date'])
        globals()[f"CO2_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"CO2_{the_date}_EOI_Group_{i}"]['date'])
        globals()[f"NO_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"NO_{the_date}_EOI_Group_{i}"]['date'])
        
        # Get the brakes for EOI of that Group
        brakes = get_brakes(globals()[f"NO2_{the_date}_EOI_Group_{i}"])
        brakes = [x for x in brakes if str(x) != 'NaT']  # Sometimes they come with an extra, so cut it out
        brakes = [*set(brakes)]  # Need to exclude repeated values due to seaborn output format
        brakes.sort()  # Corrects the order (I guess)
        idx_no2 = []
        idx_no2.append(0)
        idx_ufp = []
        idx_ufp.append(0)
        idx_o3 = []
        idx_o3.append(0)
        idx_co = []
        idx_co.append(0)
        idx_co2 = []
        idx_co2.append(0)
        idx_no = []
        idx_no.append(0)
        u = 0

        if len(brakes) > 2:  # Check if subgroups are needed (EOI detected more than once)
            # Creates the Heatmap variables and annotations per subgroup
            for j in range(1, len(brakes)):

                globals()[f"NO2_{the_date}_EOI_Group_{i}"].sort_values(by='date', inplace=True, ignore_index=True)  # Re-order dataframe to be according to date
                idx_no2.append(int(globals()[f"NO2_{the_date}_EOI_Group_{i}"][globals()[f"NO2_{the_date}_EOI_Group_{i}"]['date'] == brakes[j]].index[0]))  # Get the breaks corresponding indexes
                globals()[f"NO2_{the_date}_EOI_Group_{i}_Sub{j}"] = globals()[f"NO2_{the_date}_EOI_Group_{i}"].iloc[idx_no2[j-1]:idx_no2[j], :]  # Slices dataframe according to breaks
                                
                globals()[f"UFP_{the_date}_EOI_Group_{i}"].sort_values(by='date', inplace = True, ignore_index=True)  # Re-order dataframe to be according to date
                idx_ufp.append(int(globals()[f"UFP_{the_date}_EOI_Group_{i}"][globals()[f"UFP_{the_date}_EOI_Group_{i}"]['date'] == brakes[j]].index[0]))  # Get the breaks corresponding indexes
                globals()[f"UFP_{the_date}_EOI_Group_{i}_Sub{j}"] = globals()[f"UFP_{the_date}_EOI_Group_{i}"].iloc[idx_ufp[j-1]:idx_ufp[j], :]  # Slices data frame according to breaks
                               
                globals()[f"O3_{the_date}_EOI_Group_{i}"].sort_values(by='date', inplace = True, ignore_index=True)  # Re-order dataframe to be according to date
                idx_o3.append(int(globals()[f"O3_{the_date}_EOI_Group_{i}"][globals()[f"O3_{the_date}_EOI_Group_{i}"]['date'] == brakes[j]].index[0]))  # Get the breaks corresponding indexes
                globals()[f"O3_{the_date}_EOI_Group_{i}_Sub{j}"] = globals()[f"O3_{the_date}_EOI_Group_{i}"].iloc[idx_o3[j-1]:idx_o3[j], :]  # Slices data frame according to breaks
                
                globals()[f"CO_{the_date}_EOI_Group_{i}"].sort_values(by='date', inplace = True, ignore_index=True)  # Re-order dataframe to be according to date
                idx_co.append(int(globals()[f"CO_{the_date}_EOI_Group_{i}"][globals()[f"CO_{the_date}_EOI_Group_{i}"]['date'] == brakes[j]].index[0]))  # Get the breaks corresponding indexes
                globals()[f"CO_{the_date}_EOI_Group_{i}_Sub{j}"] = globals()[f"CO_{the_date}_EOI_Group_{i}"].iloc[idx_co[j-1]:idx_co[j], :]  # Slices data frame according to breaks
                
                globals()[f"CO2_{the_date}_EOI_Group_{i}"].sort_values(by='date', inplace = True, ignore_index=True)  # Re-order dataframe to be acCO2rding to date
                idx_co2.append(int(globals()[f"CO2_{the_date}_EOI_Group_{i}"][globals()[f"CO2_{the_date}_EOI_Group_{i}"]['date'] == brakes[j]].index[0]))  # Get the breaks CO2rresponding indexes
                globals()[f"CO2_{the_date}_EOI_Group_{i}_Sub{j}"] = globals()[f"CO2_{the_date}_EOI_Group_{i}"].iloc[idx_co2[j-1]:idx_co2[j], :]  # Slices data frame acCO2rding to breaks
                
                globals()[f"NO_{the_date}_EOI_Group_{i}"].sort_values(by='date', inplace = True, ignore_index=True)  # Re-order dataframe to be acNOrding to date
                idx_no.append(int(globals()[f"NO_{the_date}_EOI_Group_{i}"][globals()[f"NO_{the_date}_EOI_Group_{i}"]['date'] == brakes[j]].index[0]))  # Get the breaks NOrresponding indexes
                globals()[f"NO_{the_date}_EOI_Group_{i}_Sub{j}"] = globals()[f"NO_{the_date}_EOI_Group_{i}"].iloc[idx_no[j-1]:idx_no[j], :]  # Slices data frame acNOrding to breaks
                                
                globals()[f"Group_{i}_{the_date}_counts"] = globals()[f"Group_{i}_{the_date}_counts"]+1  # Auxiliary variable for plotting
                
                # Get the dates for the annotation and vertical lines
                globals()[f"dates_ibtw_{the_date}_Group_{i}_Sub{j}"] = []
                if (j % 2) == 1:
                    interval = pd.Interval(left=brakes[j-1], right=brakes[j])
                    for k in range(0,len(globals()[f"AQ_{the_date}_dates"])):
                        if globals()[f"AQ_{the_date}_dates"][k] in interval:
                            globals()[f"dates_ibtw_{the_date}_Group_{i}_Sub{j}"].append(globals()[f"AQ_{the_date}_dates"][k])

                # Build Heatmaps per subgroup

                globals()[f"NO2_{the_date}_EOI_Group_{i}_Sub{j}"]['date'] = pd.to_datetime(globals()[f"NO2_{the_date}_EOI_Group_{i}_Sub{j}"]['date']).dt.strftime('%H:%M:%S')
                globals()[f"NO2_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"NO2_{the_date}_EOI_Group_{i}_Sub{j}"].pivot(index="Air pollutant", columns="date", values="AQ")
                globals()[f"NO2_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"NO2_{the_date}_Heatvalues_Group_{i}_Sub{j}"][globals()[f"NO2_{the_date}_Heatvalues_Group_{i}_Sub{j}"].columns].astype(float)  # or int

                globals()[f"UFP_{the_date}_EOI_Group_{i}_Sub{j}"]['date'] = pd.to_datetime(globals()[f"UFP_{the_date}_EOI_Group_{i}_Sub{j}"]['date']).dt.strftime('%H:%M:%S')
                globals()[f"UFP_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"UFP_{the_date}_EOI_Group_{i}_Sub{j}"].pivot(index="Air pollutant", columns="date", values="AQ")
                globals()[f"UFP_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"UFP_{the_date}_Heatvalues_Group_{i}_Sub{j}"][globals()[f"UFP_{the_date}_Heatvalues_Group_{i}_Sub{j}"].columns].astype(float)  # or int

                globals()[f"O3_{the_date}_EOI_Group_{i}_Sub{j}"]['date'] = pd.to_datetime(globals()[f"O3_{the_date}_EOI_Group_{i}_Sub{j}"]['date']).dt.strftime('%H:%M:%S')
                globals()[f"O3_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"O3_{the_date}_EOI_Group_{i}_Sub{j}"].pivot(index="Air pollutant", columns="date", values="AQ")
                globals()[f"O3_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"O3_{the_date}_Heatvalues_Group_{i}_Sub{j}"][globals()[f"O3_{the_date}_Heatvalues_Group_{i}_Sub{j}"].columns].astype(float)  # or int

                globals()[f"CO_{the_date}_EOI_Group_{i}_Sub{j}"]['date'] = pd.to_datetime(globals()[f"CO_{the_date}_EOI_Group_{i}_Sub{j}"]['date']).dt.strftime('%H:%M:%S')
                globals()[f"CO_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"CO_{the_date}_EOI_Group_{i}_Sub{j}"].pivot(index="Air pollutant", columns="date", values="AQ")
                globals()[f"CO_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"CO_{the_date}_Heatvalues_Group_{i}_Sub{j}"][globals()[f"CO_{the_date}_Heatvalues_Group_{i}_Sub{j}"].columns].astype(float)  # or int

                globals()[f"CO2_{the_date}_EOI_Group_{i}_Sub{j}"]['date'] = pd.to_datetime(globals()[f"CO2_{the_date}_EOI_Group_{i}_Sub{j}"]['date']).dt.strftime('%H:%M:%S')
                globals()[f"CO2_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"CO2_{the_date}_EOI_Group_{i}_Sub{j}"].pivot(index="Air pollutant", columns="date", values="AQ")
                globals()[f"CO2_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"CO2_{the_date}_Heatvalues_Group_{i}_Sub{j}"][globals()[f"CO2_{the_date}_Heatvalues_Group_{i}_Sub{j}"].columns].astype(float)  # or int

                globals()[f"NO_{the_date}_EOI_Group_{i}_Sub{j}"]['date'] = pd.to_datetime(globals()[f"NO_{the_date}_EOI_Group_{i}_Sub{j}"]['date']).dt.strftime('%H:%M:%S')
                globals()[f"NO_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"NO_{the_date}_EOI_Group_{i}_Sub{j}"].pivot(index="Air pollutant", columns="date", values="AQ")
                globals()[f"NO_{the_date}_Heatvalues_Group_{i}_Sub{j}"] = globals()[f"NO_{the_date}_Heatvalues_Group_{i}_Sub{j}"][globals()[f"NO_{the_date}_Heatvalues_Group_{i}_Sub{j}"].columns].astype(float)  # or int
                
                # What is what
                if (j % 2) == 1:
                    if "Possible" in globals()[f"NO2_{the_date}_EOI_Group_{i}"].iloc[0, 3]:  # EOI column
                        globals()[f"EOI_aux_{the_date}_Group_{i}"] = globals()[f"NO2_{the_date}_EOI_Group_{i}"].iloc[0, 3]
                        print("AQ_{the_date}_Heatmap_Group_{i}_Sub{j}".format(the_date=the_date, i=i, j=j), " is related to {EOI}".format(EOI=globals()[f"EOI_aux_{the_date}_Group_{i}"]))

        else:  # Rinse and repeat (but without subgroups)
            globals()[f"dates_ibtw_{the_date}_Group_{i}"] = []
            while u < len(brakes):
                interval = pd.Interval(left=brakes[u], right=brakes[u+1])
                u = u+2
                for k in range(0,len(globals()[f"AQ_{the_date}_dates"])):
                    if globals()[f"AQ_{the_date}_dates"][k] in interval:
                        globals()[f"dates_ibtw_{the_date}_Group_{i}"].append(globals()[f"AQ_{the_date}_dates"][k])

            # Building heatmaps per group:

            globals()[f"NO2_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"NO2_{the_date}_EOI_Group_{i}"]['date']).dt.strftime('%H:%M:%S')
            globals()[f"NO2_{the_date}_Heatvalues_Group_{i}"] = globals()[f"NO2_{the_date}_EOI_Group_{i}"].pivot(index="Air pollutant", columns="date", values="AQ")
            globals()[f"NO2_{the_date}_Heatvalues_Group_{i}"] = globals()[f"NO2_{the_date}_Heatvalues_Group_{i}"][globals()[f"NO2_{the_date}_Heatvalues_Group_{i}"].columns].astype(float)  # or int

            globals()[f"UFP_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"UFP_{the_date}_EOI_Group_{i}"]['date']).dt.strftime('%H:%M:%S')
            globals()[f"UFP_{the_date}_Heatvalues_Group_{i}"] = globals()[f"UFP_{the_date}_EOI_Group_{i}"].pivot(index="Air pollutant", columns="date", values="AQ")
            globals()[f"UFP_{the_date}_Heatvalues_Group_{i}"] = globals()[f"UFP_{the_date}_Heatvalues_Group_{i}"][globals()[f"UFP_{the_date}_Heatvalues_Group_{i}"].columns].astype(float)  # or int

            globals()[f"O3_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"O3_{the_date}_EOI_Group_{i}"]['date']).dt.strftime('%H:%M:%S')
            globals()[f"O3_{the_date}_Heatvalues_Group_{i}"] = globals()[f"O3_{the_date}_EOI_Group_{i}"].pivot(index="Air pollutant", columns="date", values="AQ")
            globals()[f"O3_{the_date}_Heatvalues_Group_{i}"] = globals()[f"O3_{the_date}_Heatvalues_Group_{i}"][globals()[f"O3_{the_date}_Heatvalues_Group_{i}"].columns].astype(float)  # or int

            globals()[f"CO_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"CO_{the_date}_EOI_Group_{i}"]['date']).dt.strftime('%H:%M:%S')
            globals()[f"CO_{the_date}_Heatvalues_Group_{i}"] = globals()[f"CO_{the_date}_EOI_Group_{i}"].pivot(index="Air pollutant", columns="date", values="AQ")
            globals()[f"CO_{the_date}_Heatvalues_Group_{i}"] = globals()[f"CO_{the_date}_Heatvalues_Group_{i}"][globals()[f"CO_{the_date}_Heatvalues_Group_{i}"].columns].astype(float)  # or int

            globals()[f"CO2_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"CO2_{the_date}_EOI_Group_{i}"]['date']).dt.strftime('%H:%M:%S')
            globals()[f"CO2_{the_date}_Heatvalues_Group_{i}"] = globals()[f"CO2_{the_date}_EOI_Group_{i}"].pivot(index="Air pollutant", columns="date", values="AQ")
            globals()[f"CO2_{the_date}_Heatvalues_Group_{i}"] = globals()[f"CO2_{the_date}_Heatvalues_Group_{i}"][globals()[f"CO2_{the_date}_Heatvalues_Group_{i}"].columns].astype(float)  # or int

            globals()[f"NO_{the_date}_EOI_Group_{i}"]['date'] = pd.to_datetime(globals()[f"NO_{the_date}_EOI_Group_{i}"]['date']).dt.strftime('%H:%M:%S')
            globals()[f"NO_{the_date}_Heatvalues_Group_{i}"] = globals()[f"NO_{the_date}_EOI_Group_{i}"].pivot(index="Air pollutant", columns="date", values="AQ")
            globals()[f"NO_{the_date}_Heatvalues_Group_{i}"] = globals()[f"NO_{the_date}_Heatvalues_Group_{i}"][globals()[f"NO_{the_date}_Heatvalues_Group_{i}"].columns].astype(float)  # or int
            
            # What is what
            if "Possible" in globals()[f"NO2_{the_date}_EOI_Group_{i}"].iloc[0, 3]:  # EOI column
                globals()[f"EOI_aux_{the_date}_Group_{i}"] = globals()[f"NO2_{the_date}_EOI_Group_{i}"].iloc[0, 3]
                print("AQ_{the_date}_Heatmap_Group_{i}".format(the_date=the_date,i=i), " is related to {EOI}".format(EOI=globals()[f"EOI_aux_{the_date}_Group_{i}"]))


# #### Plotting AQ groups and subgroups (Seaborn method):
print("")
print("Plotting AQ data heatmaps...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]
    for h in range(len(globals()[f"AQ_{the_date}_groups"])):
    
        if globals()[f"Group_{h}_{the_date}_counts"] == 1:

            plt.figure(figsize=(16, 5))
            
            fig, (axs1, axs2, axs3, axs4, axs5, axs6) = plt.subplots(6, 1, sharex=True) # The number of "axs" is related to # of polutants
            fig.suptitle('Change in Air Pollutant Concentration During ' + str(globals()[f"EOI_aux_{the_date}_Group_{h}"]) + ' event', fontsize=16, weight='bold')
            fig.text(x=0.5, y=0.9, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center', va='bottom')
            ax1 = sns.heatmap(globals()[f"NO2_{the_date}_Heatvalues_Group_{h}"], vmin=1, vmax=100, cmap=seaborn_cmap, ax=axs1).invert_yaxis()
            cbar_no2 = axs1.collections[0].colorbar
            cbar_no2.set_label('(ppb)', labelpad=-45)
            axs1.set(xlabel="", ylabel="")
            ax2 = sns.heatmap(globals()[f"UFP_{the_date}_Heatvalues_Group_{h}"], vmin=1, vmax=100000, cmap=seaborn_cmap, ax=axs2).invert_yaxis()
            cbar_ufp = axs2.collections[0].colorbar
            cbar_ufp.set_label('(#/cm\u00b3)', labelpad=-62)
            axs2.set(xlabel="", ylabel="")
            ax3 = sns.heatmap(globals()[f"O3_{the_date}_Heatvalues_Group_{h}"], vmin=1, vmax=100, cmap=seaborn_cmap, ax=axs3).invert_yaxis()
            cbar_o3 = axs3.collections[0].colorbar
            cbar_o3.set_label('(ppb)', labelpad=-45)
            axs3.set(xlabel="", ylabel="")
            ax4 = sns.heatmap(globals()[f"CO_{the_date}_Heatvalues_Group_{h}"], vmin=1, vmax=10, cmap=seaborn_cmap, ax=axs4).invert_yaxis()
            cbar_co = axs4.collections[0].colorbar
            cbar_co.set_label('(ppm)', labelpad=-40)
            axs4.set(xlabel="", ylabel="")
            ax5 = sns.heatmap(globals()[f"CO2_{the_date}_Heatvalues_Group_{h}"], vmin=1, vmax=1000, cmap=seaborn_cmap, ax=axs5).invert_yaxis()
            cbar_co2 = axs5.collections[0].colorbar
            cbar_co2.set_label('(ppm)', labelpad=-52)
            axs5.set(xlabel="", ylabel="")
            ax6 = sns.heatmap(globals()[f"NO_{the_date}_Heatvalues_Group_{h}"], vmin=1, vmax=100, cmap=seaborn_cmap, ax=axs6).invert_yaxis()
            cbar_no = axs6.collections[0].colorbar
            cbar_no.set_label('(ppb)', labelpad=-45)
            axs6.set(ylabel="")
            plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)

            for v in range(0, len(globals()[f"dates_ibtw_{the_date}_Group_{h}"])):  # Trick to include vertical lines and annotations
                x = globals()[f"dates_ibtw_{the_date}_Group_{h}"][v]-pd.Timedelta(1, unit='S')
                t = pd.to_datetime(x).strftime('%H:%M:%S')
                test = difflib.get_close_matches(t, globals()[f"NO2_{the_date}_Heatvalues_Group_{h}"].columns, n=1, cutoff=0.5)
                idx = globals()[f"NO2_{the_date}_Heatvalues_Group_{h}"].columns.get_loc(test[0])
                axs1.axvline(idx, linewidth=2, color='r')
                axs2.axvline(idx, linewidth=2, color='r')
                axs3.axvline(idx, linewidth=2, color='r')
                axs4.axvline(idx, linewidth=2, color='r')
                axs5.axvline(idx, linewidth=2, color='r')
                axs6.axvline(idx, linewidth=2, color='r')
           
            eoi_fig = str(globals()[f"EOI_aux_{the_date}_Group_{h}"])
            plt.savefig(output_folder_fig+'Seaborn_AQ_TrueConc_Out_{the_date}_Group_{h}_{eoi_fig}.png'.format(the_date=the_date, h=h, eoi_fig=eoi_fig), bbox_inches="tight", dpi=500, transparent=False)
            plt.close()

        else:

            for i in range(globals()[f"Group_{h}_{the_date}_counts"]):
                if (i % 2) == 1:
                    plt.figure(figsize=(16, 5))

                    fig, (axs1, axs2, axs3, axs4, axs5, axs6) = plt.subplots(6, 1, sharex=True)
                    fig.suptitle('Change in Air Pollutant Concentration During ' + str(globals()[f"EOI_aux_{the_date}_Group_{h}"]) + ' event', fontsize=16, weight='bold')
                    fig.text(x=0.5, y=0.9, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center', va='bottom')
                    ax1 = sns.heatmap(globals()[f"NO2_{the_date}_Heatvalues_Group_{h}_Sub{i}"], vmin=1, vmax=100, cmap=seaborn_cmap, ax=axs1).invert_yaxis()
                    cbar_no2 = axs1.collections[0].colorbar
                    cbar_no2.set_label('(ppb)', labelpad=-45)
                    axs1.set(xlabel="", ylabel="")
                    ax2 = sns.heatmap(globals()[f"UFP_{the_date}_Heatvalues_Group_{h}_Sub{i}"], vmin=1, vmax=100000, cmap=seaborn_cmap, ax=axs2).invert_yaxis()
                    cbar_ufp = axs2.collections[0].colorbar
                    cbar_ufp.set_label('(#/cm\u00b3)', labelpad=-62)
                    axs2.set(xlabel="", ylabel="")
                    ax3 = sns.heatmap(globals()[f"O3_{the_date}_Heatvalues_Group_{h}_Sub{i}"], vmin=1, vmax=100, cmap=seaborn_cmap, ax=axs3).invert_yaxis()
                    cbar_o3 = axs3.collections[0].colorbar
                    cbar_o3.set_label('(ppb)', labelpad=-45)
                    axs3.set(xlabel="", ylabel="")
                    ax4 = sns.heatmap(globals()[f"CO_{the_date}_Heatvalues_Group_{h}_Sub{i}"], vmin=1, vmax=10, cmap=seaborn_cmap, ax=axs4).invert_yaxis()
                    cbar_co = axs4.collections[0].colorbar
                    cbar_co.set_label('(ppm)', labelpad=-40)
                    axs4.set(xlabel="", ylabel="")
                    ax5 = sns.heatmap(globals()[f"CO2_{the_date}_Heatvalues_Group_{h}_Sub{i}"], vmin=1, vmax=1000, cmap=seaborn_cmap, ax=axs5).invert_yaxis()
                    cbar_co2 = axs5.collections[0].colorbar
                    cbar_co2.set_label('(ppm)', labelpad=-52)
                    axs5.set(xlabel="", ylabel="")
                    ax6 = sns.heatmap(globals()[f"NO_{the_date}_Heatvalues_Group_{h}_Sub{i}"], vmin=1, vmax=100, cmap=seaborn_cmap, ax=axs6).invert_yaxis()
                    cbar_no = axs6.collections[0].colorbar
                    cbar_no.set_label('(ppb)', labelpad=-45)
                    axs6.set(ylabel="")
                    plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)
                    
                    for v in range(0, len(globals()[f"dates_ibtw_{the_date}_Group_{h}_Sub{i}"])):  # Trick to include vertical lines and annotations
                        x = globals()[f"dates_ibtw_{the_date}_Group_{h}_Sub{i}"][v]-pd.Timedelta(1, unit='S')
                        t = pd.to_datetime(x).strftime('%H:%M:%S')
                        test = difflib.get_close_matches(t, globals()[f"NO2_{the_date}_Heatvalues_Group_{h}_Sub{i}"].columns, n=1, cutoff=0.5)
                        idx = globals()[f"NO2_{the_date}_Heatvalues_Group_{h}_Sub{i}"].columns.get_loc(test[0])
                        axs1.axvline(idx, linewidth=2, color='r')
                        axs2.axvline(idx, linewidth=2, color='r')
                        axs3.axvline(idx, linewidth=2, color='r')
                        axs4.axvline(idx, linewidth=2, color='r')
                        axs5.axvline(idx, linewidth=2, color='r')
                        axs6.axvline(idx, linewidth=2, color='r')

                    eoi_fig = str(globals()[f"EOI_aux_{the_date}_Group_{h}"])
                    plt.savefig(output_folder_fig+'Seaborn_AQ_TrueConc_Out_{the_date}_Group_{h}_Sub{i}_{eoi_fig}.png'.format(the_date=the_date, h=h, i=i, eoi_fig=eoi_fig), bbox_inches="tight", dpi=500, transparent=False)
                    plt.close()


# #### Plotting LOG groups (Seaborn method):
print("Plotting AQ data in log scale...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]
    for h in range(len(globals()[f"AQ_{the_date}_groups"])):
    
        if globals()[f"Group_{h}_{the_date}_counts"] == 1:

            plt.figure(figsize=(16, 5))
            
            fig, (axs1, axs2, axs3, axs4, axs5, axs6) = plt.subplots(6, 1, sharex=True)
            fig.suptitle('Change in Air Pollutant Concentration During ' + str(globals()[f"EOI_aux_{the_date}_Group_{h}"]) + ' event', fontsize=16, weight='bold')
            fig.text(x=0.5, y=0.9, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center', va='bottom')
            ax1 = sns.heatmap(globals()[f"NO2_{the_date}_Heatvalues_Group_{h}"], norm=LogNorm(vmin=0.001, vmax=100), cmap=seaborn_cmap, ax=axs1).invert_yaxis()
            cbar_no2 = axs1.collections[0].colorbar
            cbar_no2.set_label('(ppb)')
            cbar_no2.ax.tick_params(labelsize=5)
            axs1.set(xlabel="", ylabel="")
            ax2 = sns.heatmap(globals()[f"UFP_{the_date}_Heatvalues_Group_{h}"], norm=LogNorm(vmin=1, vmax=100000), cmap=seaborn_cmap, ax=axs2).invert_yaxis()
            cbar_ufp = axs2.collections[0].colorbar
            cbar_ufp.set_label('(#/cm\u00b3)')
            cbar_ufp.ax.tick_params(labelsize=5)
            axs2.set(xlabel="", ylabel="")
            ax3 = sns.heatmap(globals()[f"O3_{the_date}_Heatvalues_Group_{h}"], norm=LogNorm(vmin=0.001, vmax=100), cmap=seaborn_cmap, ax=axs3).invert_yaxis()
            cbar_o3 = axs3.collections[0].colorbar
            cbar_o3.set_label('(ppb)')
            cbar_o3.ax.tick_params(labelsize=5)
            axs3.set(xlabel="", ylabel="")
            ax4 = sns.heatmap(globals()[f"CO_{the_date}_Heatvalues_Group_{h}"], norm=LogNorm(vmin=0.0001, vmax=10), cmap=seaborn_cmap, ax=axs4).invert_yaxis()
            cbar_co = axs4.collections[0].colorbar
            cbar_co.set_label('(ppm)') 
            cbar_co.ax.tick_params(labelsize=5)
            axs4.set(xlabel="", ylabel="")
            ax5 = sns.heatmap(globals()[f"CO2_{the_date}_Heatvalues_Group_{h}"], norm=LogNorm(vmin=0.01, vmax=1000), cmap=seaborn_cmap, ax=axs5).invert_yaxis()
            cbar_co2 = axs5.collections[0].colorbar
            cbar_co2.set_label('(ppm)')
            cbar_co2.ax.tick_params(labelsize=5)
            axs5.set(xlabel="", ylabel="")
            ax6 = sns.heatmap(globals()[f"NO_{the_date}_Heatvalues_Group_{h}"], norm=LogNorm(vmin=0.001, vmax=100), cmap=seaborn_cmap, ax=axs6).invert_yaxis()
            cbar_no = axs6.collections[0].colorbar
            cbar_no.set_label('(ppb)')
            cbar_no.ax.tick_params(labelsize=5)
            axs6.set(ylabel="")
            plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)

            for v in range(0,len(globals()[f"dates_ibtw_{the_date}_Group_{h}"])):  # Trick to include vertical lines and annotations
                x = globals()[f"dates_ibtw_{the_date}_Group_{h}"][v]-pd.Timedelta(1, unit='S')
                t = pd.to_datetime(x).strftime('%H:%M:%S')
                test = difflib.get_close_matches(t, globals()[f"NO2_{the_date}_Heatvalues_Group_{h}"].columns, n=1, cutoff=0.5)
                idx = globals()[f"NO2_{the_date}_Heatvalues_Group_{h}"].columns.get_loc(test[0])
                axs1.axvline(idx, linewidth=2, color='r')
                axs2.axvline(idx, linewidth=2, color='r')
                axs3.axvline(idx, linewidth=2, color='r')
                axs4.axvline(idx, linewidth=2, color='r')
                axs5.axvline(idx, linewidth=2, color='r')
                axs6.axvline(idx, linewidth=2, color='r')
           
            eoi_fig = str(globals()[f"EOI_aux_{the_date}_Group_{h}"])
            plt.savefig(output_folder_fig+'Seaborn_LogNorm_AQ_TrueConc_Out_{the_date}_Group_{h}_{eoi_fig}.png'.format(the_date=the_date, h=h, eoi_fig=eoi_fig), bbox_inches="tight", dpi=500, transparent=False)
            plt.close()

        else:

            for i in range(globals()[f"Group_{h}_{the_date}_counts"]):
                if (i % 2) == 1:
                    plt.figure(figsize=(16, 5))

                    fig, (axs1, axs2, axs3, axs4, axs5, axs6) = plt.subplots(6, 1, sharex=True)
                    fig.suptitle('Change in Air Pollutant Concentration During ' + str(globals()[f"EOI_aux_{the_date}_Group_{h}"]) + ' event', fontsize=16, weight='bold')
                    fig.text(x=0.5, y=0.9, s='EOI detection indicated by red line(s)', fontsize=12, alpha=0.75, ha='center', va='bottom')
                    ax1 = sns.heatmap(globals()[f"NO2_{the_date}_Heatvalues_Group_{h}_Sub{i}"], norm=LogNorm(vmin=0.001, vmax=100), cmap=seaborn_cmap, ax=axs1).invert_yaxis()
                    cbar_no2 = axs1.collections[0].colorbar
                    cbar_no2.set_label('(ppb)')
                    cbar_no2.ax.tick_params(labelsize=5)
                    axs1.set(xlabel="", ylabel="")
                    ax2 = sns.heatmap(globals()[f"UFP_{the_date}_Heatvalues_Group_{h}_Sub{i}"], norm=LogNorm(vmin=1, vmax=100000), cmap=seaborn_cmap, ax=axs2).invert_yaxis()
                    cbar_ufp = axs2.collections[0].colorbar
                    cbar_ufp.set_label('(#/cm\u00b3)')
                    cbar_ufp.ax.tick_params(labelsize=5)
                    axs2.set(xlabel="", ylabel="")
                    ax3 = sns.heatmap(globals()[f"O3_{the_date}_Heatvalues_Group_{h}_Sub{i}"], norm=LogNorm(vmin=0.001, vmax=100), cmap=seaborn_cmap, ax=axs3).invert_yaxis()
                    cbar_o3 = axs3.collections[0].colorbar
                    cbar_o3.set_label('(ppb)')
                    cbar_o3.ax.tick_params(labelsize=5)
                    axs3.set(xlabel="", ylabel="")
                    ax4 = sns.heatmap(globals()[f"CO_{the_date}_Heatvalues_Group_{h}_Sub{i}"], norm=LogNorm(vmin=0.0001, vmax=10), cmap=seaborn_cmap, ax=axs4).invert_yaxis()
                    cbar_co = axs4.collections[0].colorbar
                    cbar_co.set_label('(ppm)')
                    cbar_co.ax.tick_params(labelsize=5)
                    axs4.set(xlabel="", ylabel="")
                    ax5 = sns.heatmap(globals()[f"CO2_{the_date}_Heatvalues_Group_{h}_Sub{i}"], norm=LogNorm(vmin=0.01, vmax=1000), cmap=seaborn_cmap, ax=axs5).invert_yaxis()
                    cbar_co2 = axs5.collections[0].colorbar
                    cbar_co2.set_label('(ppm)')
                    cbar_co2.ax.tick_params(labelsize=5)
                    axs5.set(xlabel="", ylabel="")
                    ax6 = sns.heatmap(globals()[f"NO_{the_date}_Heatvalues_Group_{h}_Sub{i}"], norm=LogNorm(vmin=0.001, vmax=100), cmap=seaborn_cmap, ax=axs6).invert_yaxis()
                    cbar_no = axs6.collections[0].colorbar
                    cbar_no.set_label('(ppb)')
                    cbar_no.ax.tick_params(labelsize=5)
                    axs6.set(ylabel="")
                    plt.xlabel('Date {the_date}'.format(the_date=the_date), labelpad=10)
                    
                    for v in range(0,len(globals()[f"dates_ibtw_{the_date}_Group_{h}_Sub{i}"])):  # Trick to include vertical lines and annotations
                        x = globals()[f"dates_ibtw_{the_date}_Group_{h}_Sub{i}"][v]-pd.Timedelta(1, unit='S')
                        t = pd.to_datetime(x).strftime('%H:%M:%S')
                        test = difflib.get_close_matches(t, globals()[f"NO2_{the_date}_Heatvalues_Group_{h}_Sub{i}"].columns, n=1, cutoff=0.5)
                        idx = globals()[f"NO2_{the_date}_Heatvalues_Group_{h}_Sub{i}"].columns.get_loc(test[0])
                        axs1.axvline(idx, linewidth=2, color = 'r')
                        axs2.axvline(idx, linewidth=2, color = 'r')
                        axs3.axvline(idx, linewidth=2, color = 'r')
                        axs4.axvline(idx, linewidth=2, color = 'r')
                        axs5.axvline(idx, linewidth=2, color = 'r')
                        axs6.axvline(idx, linewidth=2, color = 'r')

                    eoi_fig = str(globals()[f"EOI_aux_{the_date}_Group_{h}"])
                    plt.savefig(output_folder_fig+'Seaborn_LogNorm_AQ_TrueConc_Out_{the_date}_Group_{h}_Sub{i}_{eoi_fig}.png'.format(the_date=the_date, h=h, i=i, eoi_fig=eoi_fig), bbox_inches="tight", dpi=500, transparent=False)
                    plt.close()


'''
## Plotting TimeSeries for pollutants with shades indicating events
# Creates the shades intervals for this timeseries
print("Plotting AQ data in timeseries format (all events)...")
for g in range(len(sampling_dates)):
    the_date = sampling_dates[g]
    globals()[f"AQ_{the_date}_intervals"] = create_shades(globals()[f"AQ_{the_date}_EOIs"].to_dict('series'))
    # Get the EOI list needed to plot the figures
    globals()[f"AQ_{the_date}_onlyEOIs"] = eoi_list(globals()[f"AQ_{the_date}_EOIs"])
    # Adjusting the date for plotting
    globals()[f"AQ_{the_date}_EOIs"]['date'] = pd.to_datetime(globals()[f"AQ_{the_date}_EOIs"]['date'])

    # Plotting the entire timeseries + shades of day of interest (for NOX and O3)
    plot_lines = []
    figure = plt.figure(figsize=(16,5))
    axes = figure.add_subplot(1, 1, 1)
    idx = globals()[f"AQ_{the_date}_EOIs"].columns.get_loc("EOI")
    idx_2 = globals()[f"AQ_{the_date}_EOIs"].columns.get_loc("date")

    l1, = axes.plot(globals()[f"AQ_{the_date}_EOIs"]['date'], globals()[f"AQ_{the_date}_EOIs"]['NO2 (ppb)'], color='black', zorder=4)
    l2, = axes.plot(globals()[f"AQ_{the_date}_EOIs"]['date'], globals()[f"AQ_{the_date}_EOIs"]['NO (ppb)'], color='gray', zorder=4)
    l3, = axes.plot(globals()[f"AQ_{the_date}_EOIs"]['date'], globals()[f"AQ_{the_date}_EOIs"]['O3 (ppb)'], color='red', zorder=4)
    plot_lines.append([l1, l2, l3])
    legend1 = plt.legend(plot_lines[0], ["NO2", "NO", "O3"], loc='upper left')

    for j in range(0,len(globals()[f"AQ_{the_date}_onlyEOIs"])):
        for i in range(0,len(globals()[f"AQ_{the_date}_EOIs"])):
            if globals()[f"AQ_{the_date}_EOIs"].iloc[i,idx] == globals()[f"AQ_{the_date}_onlyEOIs"][j]:
                if "Possible" in globals()[f"AQ_{the_date}_EOIs"].iloc[i,idx]:
                    plt.axvline(globals()[f"AQ_{the_date}_EOIs"].iloc[i,idx_2], label=globals()[f"AQ_{the_date}_onlyEOIs"][j], color=colors[j], zorder=3)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(axes.get_xticklabels(), rotation=90)
    plt.xlabel(the_date)
    plt.ylabel('Concentration (ppb)')
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.gca().add_artist(legend1)
    plt.savefig(output_folder_fig+'Timeseries_NOXandO3_TrueConc_Out_{the_date}.png'.format(the_date=the_date), bbox_inches='tight', dpi=500, transparent=False)
    plt.close()
'''

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

