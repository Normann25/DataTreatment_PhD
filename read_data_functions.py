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

def import_data(path, parent_path, timelabel, time_format, hour):
    data_dict = {}
    files = file_list(path, parent_path)

    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(path, file), 'r') as f:
                df = pd.read_table(f, sep = '\t')

            df = df.fillna(0)
            
            if timelabel is not None:
                try:
                    df['Time'] = format_timestamps(df[timelabel], time_format, '%d/%m/%Y %H:%M:%S')
                    df['Time'] =  df['Time'] + pd.Timedelta(hours = hour)
                except KeyError:
                    pass
            
            data_dict[file.split('.')[0]] = df
        
        if file.endswith('.csv'):
            with open(os.path.join(path, file), 'r') as f:
                df = pd.read_csv(f)
            
            if timelabel is not None:
                try:
                    df['Time'] = format_timestamps(df[timelabel], time_format, '%d/%m/%Y %H:%M:%S')
                    df['Time'] =  df['Time'] + pd.Timedelta(hours = hour)
                except KeyError:
                    pass
            
            data_dict[file.split('.')[0]] = df

        if file.endswith('.CSV'):
            with open(os.path.join(path, file), 'r') as f:
                df = pd.read_csv(f, sep = ';', decimal = ',')
            
            if timelabel is not None:
                try:
                    df['Time'] = format_timestamps(df[timelabel], time_format, '%d/%m/%Y %H:%M:%S')
                    df['Time'] =  df['Time'] + pd.Timedelta(hours = hour)
                except KeyError:
                    pass
            
            data_dict[file.split('.')[0]] = df

    return data_dict

def import_PTRMS(path, parent_path):
    data = import_data(path, parent_path, None, None, None)

    new_dict = {}
    for key in data.keys():
        df = data[key]

        Timestamps = pd.to_datetime(df['AbsTime'], origin = pd.Timestamp('1899-12-30'), unit = 'D')  # format = '%d/%m/%Y %H:%M:%S'
        df['Time'] = Timestamps

        new_dict[key] = df

    return new_dict

def import_SMPS(path, parent_path, hour):
    """Read SMPS data from CSV files in the specified path."""
    files = file_list(path, parent_path)
    SMPS_files = []

    for file in files:
        SMPS_files.append(file)

    if len(SMPS_files) > 1:
        data = {}

        for file in SMPS_files:
            separations = [',', '\t']
            for separation in separations:
                try:
                    with open(os.path.join(path, file), 'r') as f:
                        df = pd.read_csv(f, sep = separation, skiprows = 52)

                    df['Time'] = format_timestamps(df['DateTime Sample Start'], '%d/%m/%Y %H:%M:%S', "%d/%m/%Y %H:%M:%S")
                    df['Time'] = df['Time'] + pd.Timedelta(hours = hour)

                    name = file.split('.')[0]
                    data[name] = df
                    
                except KeyError:
                    pass

    else:
        file = SMPS_files[0]
        separations = [',', '\t']
        for separation in separations:
            try:
                with open(os.path.join(path, file), 'r') as f:
                    df = pd.read_csv(f, sep = separation, skiprows = 52)

                df['Time'] = format_timestamps(df['DateTime Sample Start'], '%d/%m/%Y %H:%M:%S', "%d/%m/%Y %H:%M:%S")
                df['Time'] = df['Time'] + pd.Timedelta(hours = hour)

                data = df
                
            except KeyError:
                pass
        
    return data

def import_SASS(path, parent_path, hour, minute, second):
    files = file_list(path, parent_path)
    data_dict = {}

    for file in files:
        if 'SASS' in file:
            start_time = linecache.getline(os.path.join(path, file), 2).split('\t')[1]
            date = linecache.getline(os.path.join(path, file), 2).split('\t')[3]
            date = date.split(' ')[0]
            start_time = pd.to_datetime(f'{date} {start_time}') + pd.Timedelta(hours = hour, minutes = minute, seconds = second)
            
            # List to store scans 
            scans = []
            scan_number = None
            in_data_block = False
            current_data = []

            with open(os.path.join(path, file), 'r') as f:
                for line in f:
                    line = line.strip()

                    # Start of a new scan
                    if line.startswith("SCAN"):
                        parts = line.split('\t')
                        try:
                            scan_number = int(parts[1])
                        except (IndexError, ValueError):
                            scan_number = None
                        in_data_block = False  # wait for next header to begin collecting
                        continue

                    # Header line (start of scan data)
                    if line.endswith("dLog(Da)/m3"):
                        in_data_block = True
                        continue

                    # End of a scan block
                    if line.startswith("END OF SCAN"):
                        if current_data:
                            # Convert collected lines to DataFrame
                            df = pd.DataFrame(current_data, columns=[
                                'ScanTime', 'Time', 'Size', 'SpectralDensity', 'CorrectedSpectralDensity', 
                                'tau', 'tauSpectralDensity', 'CorrTauSpectralDensity',
                                'MobilitySize', 'MobilitySpectralDensity', 'CorrMobilitySpectralDensity'
                            ])
                            df['Time'] = start_time + pd.Timedelta(minutes = (scan_number-1)*10)
                            df['ScanNumber'] = scan_number
                            df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
                            df['CorrectedSpectralDensity'] = pd.to_numeric(df['CorrectedSpectralDensity'], errors='coerce')
                            df['MobilitySize'] = pd.to_numeric(df['MobilitySize'], errors='coerce')
                            df['CorrMobilitySpectralDensity'] = pd.to_numeric(df['CorrMobilitySpectralDensity'], errors='coerce')
                            df = df[['Size', 'CorrectedSpectralDensity', 'ScanNumber', 'Time', 'MobilitySize', 'CorrMobilitySpectralDensity']].dropna()
                            scans.append(df)
                            current_data = []
                        in_data_block = False
                        continue

                    # Collect data lines
                    if in_data_block:
                        parts = line.split('\t')
                        if len(parts) >= 11:
                            current_data.append(parts[:11])
                    
            if scans:
                full_df = pd.concat(scans, ignore_index=True)
                
            data_dict[file] = full_df
               
    return data_dict