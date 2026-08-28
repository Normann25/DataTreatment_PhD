#%%
import sys
sys.path.append('../')
from read_data_functions import *
from plot_functions import *
from calculations import *
from grouping import *
plt.style.use('../Style.mplstyle')
plt.rcParams['font.family'] = 'Arial'
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # suppress warnings
#%%
parent_path = '../../../Data/2026/'
paths = ['20260824_blank_UV/', '20260827_Blank_UV_85RH/']

timestamps = [['2026-08-24 12:45', '2026-08-24 18:21'],
              ['2026-08-27 08:20', '2026-08-27 13:45']]
t_zero = ['2026-08-24 13:20', '2026-08-27 08:40']
t_UV_off = ['2026-08-24 17:21', '2026-08-27 12:45']

RH = ['Dry', '85% RH']

SMPS = {}
PTRMS = {}
# AMS = {}
DAQ = {}
for path in paths:
    temp_SMPS = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp_SMPS.keys():
        SMPS[key] = temp_SMPS[key]
        # SMPS[key] = temp_SMPS[key]
    # temp_AMS = import_data(f'{parent_path}{path}AMS/', '', 't_series', '%d-%m-%Y %H:%M:%S', 0)
    # for key in temp_AMS.keys():
    #     if 'PToF' not in key:
    #         temp_AMS[key].columns = ['t_series', 'HROrg', 'HRNO3', 'HRSO4', 'HRNH4', 'HRChl', 'Ratio_H_C', 'Ratio_O_C', 
    #                         'familyCHN', 'familyCHO1', 'familyCHOgt1', 'familyCHO1N', 'familyCH', 'f43', 'f44', 'Time']
    #         temp_AMS[key].loc[temp_AMS[key]['Time'] < pd.to_datetime(t), ['Ratio_H_C', 'Ratio_O_C']] = np.nan
    #         # temp_AMS[key].loc[temp_AMS[key]['Ratio_O_C'] < -0.1, ['Ratio_H_C', 'Ratio_O_C']] = 0
    #         # temp_AMS[key].loc[temp_AMS[key]['Ratio_O_C'] > 4, ['Ratio_H_C', 'Ratio_O_C']] = 0
    #     AMS[key] = temp_AMS[key]
    temp_ptrms = import_PTRMS(f'{parent_path}{path}PTRMS/', '')
    for key in temp_ptrms.keys():
        PTRMS[key] = temp_ptrms[key]
    temp_daq = import_data(f'{parent_path}{path}DAQ/', '', 'DAQ_Timestamp_UTC', '%d-%m-%Y %H:%M:%S', 0)
    for key in temp_daq.keys():
        temp = remove_spikes(temp_daq[key], ['Temp_C'], 5)
        DAQ[key] = temp

save_path = '../../../Figures/Vanillin/2608_Blanks/'

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    if '85RH' in key:
        SMPS[key].loc[SMPS[key]['Total concentration'] <= 3, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan


RH_mask = DAQ['DataDAQ_260826']['RH_Percent'] > 10
DAQ['DataDAQ_260826'] = DAQ['DataDAQ_260826'][RH_mask]

SMPS_keys = [['20260824_blank_uv_dry_number', '20260827_blank_uv_85RH_number'],
             ['20260824_blank_uv_dry_mass', '20260827_blank_uv_85RH_mass']]
DAQ_keys = ['DataDAQ_260824', 'DataDAQ_260826']
PTRMS_keys = ['260824_blank_dry', '260827_blank_85RH']
#%%
for i, time in enumerate(timestamps):
    plot_AURA_overview(DAQ[DAQ_keys[i]], SMPS[SMPS_keys[0][i]], None, time, None, t_zero[i], RH[i], save_path)
#%%
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['20260824_blank_uv_dry_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, RH, 'Total concentration', t_zero, 1, 2, save_path)
#%%
# PTR-MS grouping of ions
for key in PTRMS_keys:
    # Identify concentration columns
    concentration_cols = [col for col in PTRMS[key].columns if col.startswith('m') and '(' in col] # The name of the time series
    smooth_data_array = GetData(PTRMS[key], concentration_cols, smooth=True, window_size=12, normalize=True)
    data_array = GetData(PTRMS[key], concentration_cols, smooth=False, window_size=50, normalize=True)

    # Compute Distance measures
    smooth_distance_matrices = ComputeTSDistance(smooth_data_array, 'p4')
    distance_matrices = ComputeTSDistance(data_array, 'p4')

    # Do clustering and plot the result
    for label, d_mat in smooth_distance_matrices.items():
        hdbscan_labels= PerformHDBSCAN(d_mat)
        PlotClusterRows(smooth_data_array, concentration_cols, hdbscan_labels, f'HDBSCAN Clustering: {label}', f'{save_path}hdbscan_clusters_{key.split('_')[0]}_{label}_smooth.jpg')
    for label, d_mat in distance_matrices.items():
        hdbscan_labels= PerformHDBSCAN(d_mat) # Element x in concentration_cols belongs to cluster i where i is element x in hdbscan_labels
        PlotClusterRows(data_array, concentration_cols, hdbscan_labels, f'HDBSCAN Clustering: {label}', f'{save_path}hdbscan_clusters_{key.split('_')[0]}_{label}_raw.jpg')