#%%
import sys
sys.path.append('../')
from read_data_functions import *
from plot_functions import *
from calculations import *
from grouping import *
plt.style.use('../Style.mplstyle')
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # suppress warnings
#%%
parent_path = '../../../Data/2026/'
paths = ['260427_Vanillin70ppb_UV_RH70/', '260428_Vanillin70ppb_UV_RH70/', 
         '260429_Vanillin70ppb_UV_RH85/', '260430_Vanillin70ppb_UV_RH85/']

timestamps = [['2026-04-27 10:36', '2026-04-27 17:28'],
              ['2026-04-28 11:07', '2026-04-28 17:38'],
              ['2026-04-29 08:52', '2026-04-29 16:45'],
              ['2026-04-30 07:58', '2026-04-30 17:56']]
t_VL_inj = ['2026-04-29 10:02', '2026-04-30 11:11']
t_zero = ['2026-04-27 12:16', '2026-04-28 12:35', '2026-04-29 11:45', '2026-04-30 12:47']
t_UV_off = ['2026-04-27 16:21', '2026-04-28 16:36', '2026-04-29 15:45', '2026-04-30 16:47']
HEPA_timestamps = [['2026-04-27 08:40', '2026-04-27 09:00'],
                   ['2026-04-28 07:55', '2026-04-28 08:15'],
                   ['2026-04-29 08:40', '2026-04-29 09:00'],
                   ['2026-04-30 08:15', '2026-04-30 08:35']]

RH = ['70% RH', '70% RH', '85% RH', '85% RH']

SMPS = {}
SMPS_raw = {}
PTRMS = {}
AMS = {}
DAQ = {}
for t, path in zip(t_zero, paths):
    temp_SMPS = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp_SMPS.keys():
        SMPS_raw[key] = temp_SMPS[key]
        temp_SMPS[key].loc[temp_SMPS[key]['Time'] < pd.to_datetime(t) + pd.Timedelta(minutes = 20), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        temp = remove_spikes_up(temp_SMPS[key], ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 20)
        SMPS[key] = temp
    if path != paths[0]:
        temp_PTR = import_PTRMS(f'{parent_path}{path}PTRMS/', '')
        for key in temp_PTR.keys():
            mask = (0 < temp_PTR[key][temp_PTR[key].keys()[5]]) & (temp_PTR[key][temp_PTR[key].keys()[5]] < 90)
            temp = temp_PTR[key][mask]
            PTRMS[key] = temp.drop(['AbsTime', 'RelTime', 'Cycle', 'CycleInFile', 'Filename'], axis = 1)
    temp_AMS = import_data(f'{parent_path}{path}AMS/', '', 't_series', '%d-%m-%Y %H:%M:%S', 0)
    for key in temp_AMS.keys():
        if 'PToF' not in key:
            temp_AMS[key].columns = ['t_series', 'HROrg', 'HRNO3', 'HRSO4', 'HRNH4', 'HRChl', 'Ratio_H_C', 'Ratio_O_C', 
                            'familyCHN', 'familyCHO1', 'familyCHOgt1', 'familyCHO1N', 'familyCH', 'f43', 'f44', 'Time']
        AMS[key] = temp_AMS[key]
    temp_daq = import_data(f'{parent_path}{path}DAQ/', '', 'DAQ_Timestamp_UTC', '%d-%m-%Y %H:%M:%S', 0)
    for key in temp_daq.keys():
        DAQ[key] = temp_daq[key]

save_path = '../../../Figures/Vanillin/2604_vanillin+UV_RH70-85/'

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    # SMPS[key] = SMPS[key].fillna(0)

SMPS_keys = [['260427_vanillin+UV_RH70_number', '260428_vanillin+UV_RH70_number', '260429_vanillin+UV_RH85_number', '260430_vanillin+UV_RH85_number'],
             ['260427_vanillin+UV_RH70_mass', '260428_vanillin+UV_RH70_mass', '260429_vanillin+UV_RH85_mass', '260430_vanillin+UV_RH85_mass']]
AMS_keys = ['260427_AMS_vanillin+UV_70RH_TS', '260428_AMS_vanillin+UV_70RH_TS', '260429_AMS_vanillin+UV_85RH_TS', '260430_AMS_vanillin+UV_85RH_TS']
PTRMS_keys = ['260428_VL+UV_70RH_initial', '260429_VL+UV_RH85_initial', '260430_VL+UV_RH85_inital']
DAQ_keys = ['DataDAQ_260427', 'DataDAQ_260428', 'DataDAQ_260429', 'DataDAQ_260430']
#%%
for i, time in enumerate(timestamps):
    fig, ax = plot_AURA_overview(DAQ[DAQ_keys[i]], SMPS[SMPS_keys[0][i]], AMS[AMS_keys[i]], time, HEPA_timestamps[i], t_zero[i], RH[i], save_path)
    ax[2].set_ylim (0, 1)
#%%
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260428_vanillin+UV_RH70_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, RH, 'Total concentration', t_zero, 2, 2, save_path)
#%%
for i, key in enumerate(AMS_keys):
    plot_AMS(AMS[key], None, t_zero[i], timestamps[i], HEPA_timestamps[i], 1, RH[i], save_path)
#%%
print(PTRMS.keys())
#%%
# PTR-MS grouping of ions
for key in ['260429_VL+UV_RH85_products', '260430_VL+UV_RH85_products']:
    # Identify concentration columns
    concentration_cols = [col for col in PTRMS[key].columns if col.startswith('m') and '(' in col] # The name of the time series
    smooth_data_array = GetData(PTRMS[key], concentration_cols, smooth=True, window_size=12)
    data_array = GetData(PTRMS[key], concentration_cols, smooth=False, window_size=50)

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
#%%
# PTR-MS O:C and H:C
PTR_merge_keys = [['260429_VL+UV_RH85_fragments', '260429_VL+UV_RH85_products'],
                  ['260430_VL+UV_RH85_fragments', '260430_VL+UV_RH85_products']]
for i, keys in enumerate(PTR_merge_keys):
    merged = pd.merge(PTRMS[keys[0]], PTRMS[keys[1]], on = 'Time', how = 'outer')
    PTRMS[f'{keys[0].split('_')[0]}_VL+UV_dry_OC-HC'] = calc_OC_HC_PTRMS(merged)

    fig, ax = vanKrevelen_ts(PTRMS[f'{keys[0].split('_')[0]}_VL+UV_dry_OC-HC'], ['Ratio_H_C', 'Ratio_O_C'], None,
                             t_zero[i+2], [t_VL_inj[i], t_UV_off[i+2]], 5/60, f'{t_UV_off[i+2].split(' ')[0]}, 85% RH')
    fig.tight_layout(pad = 0.75)
    fig.savefig(f'{save_path}{timestamps[i+2][0].split(' ')[0]}_vanKrevelen_PTRMS.jpg', dpi = 600)
#%%
for i, key in enumerate(PTRMS_keys):
    fig, ax = plot_PTRMS_decay(PTRMS[key], PTRMS[key].keys()[5], [PTRMS[key].keys()[4]], 
                               ['C$_{8}$H$_{9}$O$_{3}^{+}$', 'C$_{7}$H$_{9}$O$_{2}^{+}$'], 
                               t_zero[i+1], t_UV_off[i], timestamps[i+1][1], RH[i+1])
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+1].split(' ')[0]}_PTRMS_initial.jpg', dpi = 600)