import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../../')
from Functions import file_list, format_timestamps, calc_total, running_mean, time_filtered_conc

def read_csv(path, parent_path, timelabel, time_format):
    data_dict = {}
    files = file_list(path, parent_path)

    for file in files:
        if file.endswith('.csv') or file.endswith('.CSV'):
            separations = [',', ';']
            name = file.split('.')[0]
            for sep in separations:
                try:
                    with open(os.path.join(path, file), 'r') as f:
                        df = pd.read_csv(f, sep = sep)
                    df['Time'] = format_timestamps(df[timelabel], time_format, '%d/%m/%Y %H:%M:%S')
                    data_dict[name] = df

                except KeyError:
                    pass
                except ValueError:
                    pass

    return data_dict

path = '../../../../Courses/2026 - INAR winter school/Data/'
MION = read_csv(f'{path}MION/', '', 'Time', '%d-%b-%Y %H:%M:%S')
MION = MION['MION_Ambient_WinterSchool']

MION_NO3 = read_csv(f'{path}MION_NO3/', '', 'Time', '%Y-%m-%d %H:%M')
MION_NO3['MION_NO3_1H-avg'].to_csv(f'{path}MION_NO3_1H-avg.csv', index = False)

DMPS_temp = read_csv(f'{path}DMPS/', '', 'Date', '%Y-%m-%d %H:%M:%S')
DMPS = pd.merge(DMPS_temp['DMPS_TZS_2024'], DMPS_temp['Particle number conc_TZS_2024'], on = 'Time', how = 'outer').drop(['Date_x', 'Date_y'], axis = 1)
DMPS.to_csv(f'{path}DMPS_2024.csv', index = False)

NAIS = read_csv(f'{path}NAIS/', '', 'time', '%Y-%m-%d %H:%M:%S')
for key in NAIS.keys():
    NAIS[key] = NAIS[key].drop(['time'], axis = 1)

PSM24 = read_csv(f'{path}PSM/2024/', '', 'ScanTime', '%d-%b-%Y %H:%M:%S')
PSM25 = read_csv(f'{path}PSM/2025/', '', 'ScanTime', '%d-%b-%Y %H:%M:%S')
# Merge PSM dataframes
PSM_2024_01 = pd.DataFrame()
PSM_2024 = pd.DataFrame()
for key in PSM24.keys():
    if PSM24[key].iloc[-1]['Time'] < pd.to_datetime('2024-01-23 00:00:00'):
        PSM_2024_01 = pd.concat([PSM_2024_01, PSM24[key]], ignore_index = True)
    if pd.to_datetime('2024-01-23 00:00:00') < PSM24[key].iloc[-1]['Time']:
        PSM_2024 = pd.concat([PSM_2024, PSM24[key]], ignore_index = True)
PSM_2025 = pd.DataFrame()
for key in PSM25.keys():
    PSM_2025 = pd.concat([PSM_2025, PSM25[key]], ignore_index = True)
PSM = {'2024-01': PSM_2024_01, '2024': PSM_2024, '2025': PSM_2025}
bins = [[1.17, 1.3, 1.5, 1.7, 2.5, 3.0, 3.92],
        [1.1, 1.3, 1.5, 1.7, 2.5, 3.0],
        [1.17, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0]]

# 1-hour running mean of data
PSM_running = {}
for key, binedges in zip(PSM.keys(), bins):
    PSM[key]['Total conc'] = calc_total(PSM[key], PSM[key].keys()[1:-2], binedges)
    temp = running_mean(PSM[key], PSM[key].keys()[1:-3].to_list()+['Total conc'], 'Time', '60min', None)
    temp['Time'] = temp.index
    temp = temp.reset_index(drop = True)
    temp.to_csv(f'{path}PSM_{key}.csv', index = False)
    PSM_running[key] = temp

NAIS_running = {}
for key in NAIS.keys():
    temp = running_mean(NAIS[key], NAIS[key].keys()[:-1], 'Time', '60min', ['2023-12-31 23:59:00', '2026-01-01 00:01:00'])
    temp['Time'] = temp.index + pd.Timedelta(hours = 2)
    NAIS_running[key] = temp.reset_index(drop = True)
NAIS_running['formation_rate_2_2p3_neg'].to_csv(f'{path}NAIS_formation_rate_neg_1H-avg.csv', index = False)
NAIS_running['tvarminne_conc_utc'].to_csv(f'{path}NAIS_TZS_1H-avg.csv', index = False)

MION_running = running_mean(MION, MION.keys()[1:], 'Time', '60min', None)
MION_running['Time'] = MION_running.index
MION_running = MION_running.reset_index(drop = True)
MION_running.to_csv(f'{path}MION_TZS_1H-avg.csv', index = False)

event_dates = [['2024-03-04 23:59', '2024-03-06 23:30'],
               ['2024-03-11 23:59', '2024-03-13 23:30'],
               ['2024-03-17 23:59', '2024-03-19 23:30'],
               ['2024-04-03 23:59', '2024-04-05 23:30'],
               ['2024-04-09 23:59', '2024-04-11 23:30'],
               ['2024-04-13 23:59', '2024-04-16 23:30'],
               ['2024-04-21 23:59', '2024-04-23 23:30'],
               ['2024-05-06 23:59', '2024-05-07 23:30'], 
               ['2024-05-11 23:59', '2024-05-12 23:30'],
               ['2024-05-15 23:59', '2024-05-17 23:30'],
               ['2024-06-08 23:59', '2024-06-10 23:30'],
               ['2024-06-20 23:59', '2024-06-24 23:30'],
               ['2024-06-28 23:59', '2024-06-29 23:30'], 
               ['2024-07-04 23:59', '2024-07-05 23:30'],
               ['2024-08-11 23:59', '2024-08-13 23:30'],
               ['2024-08-18 23:59', '2024-08-20 23:30'],
               ['2024-08-24 23:59', '2024-08-26 23:30'],
               ['2024-09-09 23:59', '2024-09-11 23:30'],
               ['2024-09-14 23:59', '2024-09-16 23:30'],
               ['2024-09-18 23:59', '2024-09-21 23:30'],
               ['2024-09-28 23:59', '2024-09-30 23:30'],
               ['2024-10-02 23:59', '2024-10-03 23:30'],
               ['2024-10-13 23:59', '2024-10-16 23:30'],
               ['2024-10-25 23:59', '2024-10-28 23:30'],
               ['2024-12-01 23:59', '2024-12-03 23:30'],
               ['2024-12-09 23:59', '2024-12-12 23:30'],
               ['2024-12-24 23:59', '2024-12-27 23:30']]

merged = pd.merge(NAIS_running['formation_rate_2_2p3_neg'], MION_running, on = 'Time', how = 'outer')
merged = pd.merge(merged, MION_NO3['MION_NO3_1H-avg'], on = 'Time', how = 'outer')
merged = pd.merge(merged, DMPS, on = 'Time', how = 'outer')
event_dates_df = pd.DataFrame()
merged_keys = ['J2-2.3,-/N<2,-', 'HSO4-', '(H2SO4)HSO4-', '(H2SO4)2HSO4-', 'SA', 'IA', 'MSA']
particle_formation_events = pd.DataFrame()
for timestamps in event_dates:
    temp = time_filtered_conc(merged, merged_keys, timestamps)
    particle_formation_events = pd.concat([particle_formation_events, temp], ignore_index = True)
particle_formation_events.to_csv(f'{path}NAIS-MION_PF_dates.csv', index = False)
event_dates_df = pd.DataFrame({'Start time': [i[0] for i in event_dates],
                               'End time': [i[1] for i in event_dates]})
event_dates_df.to_csv(f'{path}Event_dates.csv', index = False)
