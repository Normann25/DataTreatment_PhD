import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import linecache
#%%
def file_list(path, parent_path):
    ParentPath = os.path.abspath(parent_path)
    if ParentPath not in sys.path:
        sys.path.insert(0, ParentPath)
    
    files = os.listdir(path)

    return files

def format_timestamps(timestamps, old_format, new_format):
    new_timestamps = []
    for timestamp in timestamps:
        old_datetime = datetime.strptime(str(timestamp), old_format)
        new_datetime = old_datetime.strftime(new_format)
        new_timestamps.append(new_datetime)
    return pd.to_datetime(new_timestamps, format=new_format)

def import_SMPS(path, parent_path, hour):
    """Read SMPS data from CSV files in the specified path."""
    files = file_list(path, parent_path)
    data_dict = {}

    for file in files:

        if 'SMPS' in file:
            separations = [',', '\t']
            name = file.split('.')[0]
            for separation in separations:
                try:
                    with open(os.path.join(path, file), 'r') as f:
                        df = pd.read_csv(f, sep = separation, skiprows = 52)

                    df['Time'] = format_timestamps(df['DateTime Sample Start'], '%d/%m/%Y %H:%M:%S', "%d/%m/%Y %H:%M:%S")
                    df['Time'] = df['Time'] + pd.Timedelta(hours = hour)

                    data_dict[name] = df

                except KeyError:
                    pass

    return data_dict