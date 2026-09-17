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

def import_AMS(paths, parent_path, hour):
    new_dict = {}

    for path in paths:
        data = import_data(f'{parent_path}{path}AMS/', '', 't_series', '%d-%m-%Y %H:%M:%S', hour)
        for key in data.keys():
            if 'PToF' not in key or 'ePToF' not in key:
                data[key].columns = ['t_series', 'HROrg', 'HRNO3', 'HRSO4', 'HRNH4', 'HRChl', 'Ratio_H_C', 'Ratio_O_C', 
                                     'familyCHN', 'familyCHO1', 'familyCHOgt1', 'familyCHO1N', 'familyCH', 'f43', 'f44', 'Time']

            new_dict[key] = data[key].drop(['t_series'], axis = 1)

    return new_dict

def import_DAQ(paths, parent_path, hour):
    new_dict = {}

    for path in paths:
        data = import_data(f'{parent_path}{path}DAQ/', '', 'DAQ_Timestamp_UTC', '%d-%m-%Y %H:%M:%S', hour)

        for key in data.keys():
            data[key]['Laser_Distance'] = (data[key]['AR500_Distance_1'] + data[key]['AR500_Distance_2']) / 2

            new_dict[key] = data[key]

    return new_dict

def import_PTRMS(paths, parent_path):
    new_dict = {}

    for path in paths:
        data = import_data(f'{parent_path}{path}PTRMS/', '', None, None, None)

        for key in data.keys():
            df = data[key]

            Timestamps = pd.to_datetime(df['AbsTime'], origin = pd.Timestamp('1899-12-30'), unit = 'D').dt.floor('s')  # format = '%d/%m/%Y %H:%M:%S'
            df['Time'] = Timestamps

            new_dict[key] = df.drop(['AbsTime', 'RelTime', 'Cycle', 'CycleInFile', 'Filename'], axis = 1)

    return new_dict

def import_SMPS(paths, parent_path, hour):
    data = {}

    for path in paths:
        """Read SMPS data from CSV files in the specified path."""
        files = file_list(f'{parent_path}{path}SMPS/', '')

        if len(files) > 1:

            for file in files:
                separations = [',', '\t']
                for separation in separations:
                    try:
                        with open(os.path.join(f'{parent_path}{path}SMPS/', file), 'r') as f:
                            df = pd.read_csv(f, sep = separation, skiprows = 52)

                        df['Time'] = format_timestamps(df['DateTime Sample Start'], '%d/%m/%Y %H:%M:%S', "%d/%m/%Y %H:%M:%S")
                        df['Time'] = df['Time'] + pd.Timedelta(hours = hour)

                        name = file.split('.')[0]
                        data[name] = df
                        
                    except KeyError:
                        pass

        else:
            file = files[0]
            separations = [',', '\t']
            for separation in separations:
                try:
                    with open(os.path.join(f'{parent_path}{path}SMPS/', file), 'r') as f:
                        df = pd.read_csv(f, sep = separation, skiprows = 52)

                    df['Time'] = format_timestamps(df['DateTime Sample Start'], '%d/%m/%Y %H:%M:%S', "%d/%m/%Y %H:%M:%S")
                    df['Time'] = df['Time'] + pd.Timedelta(hours = hour)

                    name = file.split('.')[0]
                    data[name] = df
                    
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