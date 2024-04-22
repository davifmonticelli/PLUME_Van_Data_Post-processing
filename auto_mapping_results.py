# This script was created to automatically post-process P.L.U.M.E. Van data after multiple days of sampling
# Function: Map the results from previous scripts
# Authors: Chris Kelly and Davi de Ferreyro Monticelli, iREACH group (University of British Columbia)
# Date: 2023-07-12
# Version: 1.0.0

# THIS IS A NEW FEATURE

# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "Pre-processing_PLUME_Data.py" PRIOR
# DO NOT RUN THIS SCRIPT WITHOUT RUNNING "auto_merge_data.py" PRIOR

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import sys
import io
import re
import os
from matplotlib.colors import Normalize, LogNorm
import math
import contextily as ctx
from matplotlib.patches import FancyArrow
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
file_path_txt = f'C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Maps\\Mapping_console_output.txt'

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
output_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Maps\\"
output_folder_fig = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Figures\\Maps\\"
file_names = os.listdir(source_folder)

# from Events of Interest
EOI_source_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Events of Interest\\"
EOI_file_names = os.listdir(EOI_source_folder)

# Merged AQ + GPS data (file names):
Merged_names = [file_name for file_name in file_names if re.match(r"MERGED_AQ_GPS_\d{4}_\d{2}_\d{2}.csv", file_name)]
Merged_names = [os.path.splitext(file_name)[0] for file_name in Merged_names]  # Get rid of the .csv
print("Files processed are: ", Merged_names)
# Dates:
Merged_dates = list(set([re.sub(r"MERGED_AQ_GPS_", "", var_name) for var_name in Merged_names]))
print("Dates processed are: ", Merged_dates)

# Define the list of boxes (min_latitude, max_latitude, min_longitude, max_longitude)
#boxes = [
#    (49.07862, 49.12011, -122.49412, -122.42557),
#    (49.09753, 49.10373, -122.47195, -122.46083)
#]

# Define the dictionary mapping dates to boxes
# If you know beforehand the specific boxes to zoom in for each date, declare them here
date_to_boxes = {
    '2023_05_24': [
        (49.07862, 49.12011, -122.49412, -122.42557),
        (49.09753, 49.10373, -122.47195, -122.46083)
    ],
    '2023_06_02': [
        (49.07862, 49.12011, -122.49412, -122.42557),
        (49.09753, 49.10373, -122.47195, -122.46083)
    ]
}

########################################################################################################################
# Declaring functions (helpers) used throughout the script
########################################################################################################################

# Function to convert shapefiles to Dataframes
# source: https://towardsdatascience.com/mapping-with-matplotlib-pandas-geopandas-and-basemap-in-python-d11b57ab5dac
def read_shapefile(sf):
    # Fetching the headings from the shape file
    fields = [x[0] for x in sf.fields][1:]  # Fetching the records from the shape file
    records = [list(i) for i in sf.records()]
    shps = [s.points for s in sf.shapes()]  # Converting shapefile data into pandas dataframe
    df = pd.DataFrame(columns=fields, data=records)  # Assigning the coordinates
    df = df.assign(coords=shps)
    return df

# Function to read Dashboard merged files
def read_Merged_files(date_to_run):
    # Read MERGED_AQ_GPS_XXXX_XX_XX csv files
    date = date_to_run
    file_path = source_folder
    file_to_read = f"MERGED_AQ_GPS_{date}.csv"
    sensor_data = pd.read_csv(file_path+file_to_read)

    return sensor_data

# Function to read Loops files
def read_LOOP_files(date_to_run,i):
    # Read AQ_LOOP_XXXX_XX_XX_i csv files
    date = date_to_run
    file_path = EOI_source_folder
    file_to_read = f"AQ_LOOP_{date}_{i}.csv"
    sensor_data = pd.read_csv(file_path+file_to_read)

    return sensor_data

# Function to map the loops
def create_combined_map_plots(date, files_per_date, output_folder_fig):
    num_files = len(files_per_date)
    num_rows = math.ceil(num_files / 3)
    shapefiles_folder = f"C:\\Users\\{username}\\PycharmProjects\\PLUME_Van_Data_Post-processing\\outputs\\Maps\\Shapefiles\\"
    print("")
    print(f"Processing LOOPs in: {date}")
    print("")
    pollutant_columns = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]

    for pollutant_column in pollutant_columns:
        print("Creating LOOP maps for pollutant: ", pollutant_column, ", on date: ", date)
        fig, axs = plt.subplots(num_rows, 3, figsize=(16, 5 * num_rows))

        final_min_value = 100000000000
        final_max_value = 0

        # First get the vmin and vmax accross loops and pollutants for better plotting:
        for i, files in enumerate(files_per_date):
            loop_data = read_LOOP_files(date, i + 1)
            min_value = loop_data[pollutant_column].min()
            print(f"Min. value for Loop {i}: ", min_value)
            max_value = loop_data[pollutant_column].max()
            print(f"Max. value for Loop {i}: ", max_value)
            if min_value < final_min_value:
                final_min_value = min_value
            else:
                final_min_value = final_min_value
            if max_value > final_max_value:
                final_max_value = max_value
            else:
                final_max_value = final_max_value

        for i, files in enumerate(files_per_date):
            loop_data = read_LOOP_files(date, i + 1)
            loop_data['date'] = pd.to_datetime(loop_data['date'], format='%Y-%m-%d %H:%M:%S')

            # Calculate the direction for each data point
            loop_data['next_longitude'] = loop_data['longitude'].shift(-1)
            loop_data['next_latitude'] = loop_data['latitude'].shift(-1)
            loop_data['direction'] = np.arctan2(loop_data['next_longitude'] - loop_data['longitude'],
                                                loop_data['next_latitude'] - loop_data['latitude'])
            loop_data['direction'] = loop_data['direction'].fillna(0)

            # Create a GeoDataFrame with only the starting points
            loop_geometry = [Point(xy) for xy in zip(loop_data['longitude'], loop_data['latitude'])]
            loop_data_geo = gpd.GeoDataFrame(loop_data, geometry=loop_geometry, crs="EPSG:4326")

            # data_combined = pd.concat([pd.read_csv(file) for file in files])

            # Set up the colormap
            cmap = 'viridis'  # Choose a colormap (e.g., 'viridis', 'plasma', 'coolwarm', etc.)
            vmin = final_min_value  # Minimum concentration value
            vmax = final_max_value  # Maximum concentration value
            if pollutant_column == "UFP (#/cm^3)":
                norm = LogNorm(vmin=1, vmax=vmax)  # Normalize the concentration values
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)  # Normalize the concentration values

            # Add a basemap to the plot
            ctx.add_basemap(axs.flat[i], crs=loop_data_geo.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

            deg_direction = [0]

            # Plot the arrow on a map with the colorscale
            for index, row in loop_data.iterrows():
                # Estimate distance between GPS points
                distance = math.dist([row['longitude'], row['latitude']], [row['next_longitude'], row['next_latitude']])
                # Return the direction of the arrow in degrees (saved later on .csv)
                deg_direction.append(math.degrees(row['direction']))
                # Create an arrow that points towards the direction of the movement
                # AND that scales according to speed (bigger speeds == bigger arrows)
                # For the larger perspective the arrows are blown up by a factor of 20
                arrow = FancyArrow(row['longitude'], row['latitude'], distance * np.sin(row['direction']),
                                   distance * np.cos(row['direction']),
                                   width=0, head_width=3*distance/3, head_length=3*distance, linestyle="")
                axs.flat[i].add_patch(arrow)

            # Last item of each list is "nan" so we remove it
            loop_data['direction_deg'] = deg_direction[:-1]

            # Set the aspect ratio and limits of the plot
            axs.flat[i].set_aspect('equal')
            axs.flat[i].set_xlim(loop_data['longitude'].min() - 0.001, loop_data['longitude'].max() + 0.001)
            axs.flat[i].set_ylim(loop_data['latitude'].min() - 0.001, loop_data['latitude'].max() + 0.001)

            # Color the triangles based on concentration
            colors = [plt.cm.get_cmap(cmap)(norm(c)) for c in loop_data[pollutant_column]]
            for arrow, color in zip(axs.flat[i].patches, colors):
                arrow.set_facecolor(color)

            # Create a colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axs.flat[i], shrink=0.5)
            cbar.set_label(pollutant_column)

            # Add title to the subplot with the subtitle "LOOP i"
            subplot_title = f"LOOP {i + 1}"
            axs.flat[i].set_title(subplot_title)

            # Save the GeoDataFrame as a shapefile
            warnings.simplefilter('ignore', UserWarning)
            shape_to_save = loop_data_geo
            shape_to_save['date'] = shape_to_save['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            output_shapefile_name = f"Arrows_{date}_LOOP_{i}_and_pollutant_{pollutant_column[:3]}.shp"
            shape_to_save.to_file(os.path.join(shapefiles_folder, output_shapefile_name))

        # Add an overall title to the entire figure
        fig.suptitle(f"Combined Maps for {date} and pollutant {pollutant_column[:3]}", fontsize=12, color="black")

        # Save the combined map plot for the current date
        output_file_name = f"Combined_Map_for_{date}_and_pollutant_{pollutant_column[:3]}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_fig, output_file_name), dpi=500, transparent=False)

        plt.close(fig)  # Close the figure to free memory

########################################################################################################################
# MAIN
########################################################################################################################

# Iterate over the dates and threshold values
for date in Merged_dates:
    data = read_Merged_files(date)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
    print("")
    print(f"Processing dataframe: MERGED_AQ_GPS_{date}.csv")
    print(data)

    # Calculate the direction for each data point
    data['next_longitude'] = data['longitude'].shift(-1)
    data['next_latitude'] = data['latitude'].shift(-1)
    data['direction'] = np.arctan2(data['next_longitude'] - data['longitude'], data['next_latitude'] - data['latitude'])
    data['direction'] = data['direction'].fillna(0)

    # Create a GeoDataFrame with only the starting points
    geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
    data_geo = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    # Iterate over the pollutant columns
    pollutant_columns = ["NO2 (ppb)", "UFP (#/cm^3)", "O3 (ppb)", "CO (ppm)", "CO2 (ppm)", "NO (ppb)"]

    for pollutant_column in pollutant_columns:
        print("Creating maps for pollutant: ", pollutant_column, ", on date: ", date)

        # Set up the colormap
        cmap = 'viridis'  # Choose a colormap (e.g., 'viridis', 'plasma', 'coolwarm', etc.)
        vmin = data[pollutant_column].min()  # Minimum concentration value
        vmax = data[pollutant_column].max()  # Maximum concentration value
        norm = Normalize(vmin=vmin, vmax=vmax)  # Normalize the concentration values

        print(f"Creating plot: Map_for_{date}_and_{pollutant_column[:3]}_pollutant.png")
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(16,5))

        # Add a basemap to the plot (CURRENTLY NOT WORKING AS IT SHOULD)
        ctx.add_basemap(ax, crs=data_geo.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

        deg_direction = [0]

        # Plot the arrow on a map with the colorscale
        for index, row in data.iterrows():
            # Estimate distance between GPS points
            distance = math.dist([row['longitude'], row['latitude']], [row['next_longitude'], row['next_latitude']])
            # Return the direction of the arrow in degrees (saved later on .csv)
            deg_direction.append(math.degrees(row['direction']))
            # Create an arrow that points towards the direction of the movement
            # AND that scales according to speed (bigger speeds == bigger arrows)
            # For the larger perspective the arrows are blown up by a factor of 20
            arrow = FancyArrow(row['longitude'], row['latitude'], distance*np.sin(row['direction']), distance*np.cos(row['direction']),
                               width=0, head_width=20*distance/3, head_length=20*distance, linestyle="")
            ax.add_patch(arrow)

        # Last item of each list is "nan" so we remove it
        data['direction_deg'] = deg_direction[:-1]

        # Set the aspect ratio and limits of the plot
        ax.set_aspect('equal')
        ax.set_xlim(data['longitude'].min() - 0.01, data['longitude'].max() + 0.01)
        ax.set_ylim(data['latitude'].min() - 0.01, data['latitude'].max() + 0.01)

        # Color the triangles (which indicate the Van's driving direction) based on concentration
        colors = [plt.cm.get_cmap(cmap)(norm(c)) for c in data[pollutant_column]]
        for arrow, color in zip(ax.patches, colors):
            arrow.set_facecolor(color)

        # Create a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label(pollutant_column)

        plt.suptitle("Arrows are scaled according to the mobile laboratory speed (bigger arrow == higher speeds)", fontsize=10, color="gray")
        plt.title(f'Concentration Map for {date} and {pollutant_column[:3]} with Arrow Heads')

        # Plot additional boxes
        # Check if there are specific boxes for the current date
        if date in date_to_boxes:
            date_boxes = date_to_boxes[date]

            # Plot individual boxes for the current date
            for i, box in enumerate(date_boxes):

                min_lat, max_lat, min_lon, max_lon = box
                ax.plot([min_lon, max_lon, max_lon, min_lon, min_lon],
                        [min_lat, min_lat, max_lat, max_lat, min_lat],
                        color='red', linestyle='--', linewidth=1)

                # Adjust the aspect ratio and limits of the plot to include the additional boxes
                ax.set_xlim(min(min_lon, data['longitude'].min()) - 0.01, max(max_lon, data['longitude'].max()) + 0.01)
                ax.set_ylim(min(min_lat, data['latitude'].min()) - 0.01, max(max_lat, data['latitude'].max()) + 0.01)

            # Save plot
            print(f"Saving plot: Map_for_{date}_and_{pollutant_column[:3]}_pollutant.png")
            #plt.tight_layout()
            plt.savefig(
                output_folder_fig+f"Map_for_{date}_and_{pollutant_column[:3]}_pollutant"+".png",
                dpi=500,
                transparent=False)

            # Plot individual boxes
            for i, box in enumerate(date_boxes):
                min_lat, max_lat, min_lon, max_lon = box

                # Create a new figure and axis
                fig, ax = plt.subplots(figsize=(16,5))

                # Add a basemap to the plot
                ctx.add_basemap(ax, crs=data_geo.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

                # Plot the arrow on a map with the colorscale
                for index, row in data.iterrows():
                    # Estimate distance between GPS points
                    distance = math.dist([row['longitude'], row['latitude']], [row['next_longitude'], row['next_latitude']])
                    # Return the direction of the arrow in degrees (saved later on .csv)
                    deg_direction.append(math.degrees(row['direction']))
                    # Create an arrow that points towards the direction of the movement
                    # AND that scales according to speed (bigger speeds == bigger arrows)
                    # For the smaller perspective (boxes) the arrows are blown up by a factor of 1.5
                    arrow = FancyArrow(row['longitude'], row['latitude'], distance * np.sin(row['direction']),
                                       distance * np.cos(row['direction']),
                                       width=0, head_width=1.5*distance/3, head_length=1.5*distance, linestyle="")
                    ax.add_patch(arrow)

                # Set the aspect ratio and limits of the plot to zoom into the box
                ax.set_aspect('equal')
                ax.set_xlim(min_lon - 0.001, max_lon + 0.001)
                ax.set_ylim(min_lat - 0.001, max_lat + 0.001)

                # Color the triangles based on concentration
                colors = [plt.cm.get_cmap(cmap)(norm(c)) for c in data[pollutant_column]]
                for arrow, color in zip(ax.patches, colors):
                    arrow.set_facecolor(color)

                # Create a colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
                cbar.set_label(pollutant_column)

                plt.suptitle("Arrows are scaled according to the mobile laboratory speed (bigger arrow == higher speeds)",
                             fontsize=10, color="gray")
                plt.title(f'Concentration Map for {date} and {pollutant_column[:3]} with Arrow Heads - Box {i + 1}')

                # Save plot for the individual box
                print(f"Saving plot: Map_for_{date}_and_{pollutant_column[:3]}_pollutant_Box_{i + 1}.png")
                #plt.tight_layout()
                plt.savefig(
                    output_folder_fig + f"Map_for_{date}_{pollutant_column[:3]}_pollutant_Box_{i + 1}" + ".png",
                    dpi=500,
                    transparent=False)

                plt.close(fig)  # Close the figure to free memory

        plt.close(fig)  # Close the figure for the current date

    # Get a list of CSV files for the current date containing "AQ_withGPS_LOOP" in their names
    files_per_date = [file_name for file_name in EOI_file_names if f"AQ_LOOP_{date}" in file_name]
    files_per_date = [os.path.join(EOI_source_folder, file_name) for file_name in files_per_date]

    # Create and save combined map plots for multiple files per date and pollutant
    create_combined_map_plots(date, files_per_date, output_folder_fig)
    plt.close('all')  # Close all figures after processing the current date

    # Save the new dataframe
    data.to_csv(output_folder+f"Map_for_{date}.csv", index=False)


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