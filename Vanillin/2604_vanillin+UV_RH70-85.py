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
# Paths
parent_path = '../../../Data/2026/'
paths = ['260427_Vanillin70ppb_UV_RH70/', '260428_Vanillin70ppb_UV_RH70/', 
         '260429_Vanillin70ppb_UV_RH85/', '260430_Vanillin70ppb_UV_RH85/']
save_path = '../../../Figures/Vanillin/2604_vanillin+UV_RH70-85/'

# Timestamps
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

# Exp relative humidity
RH = ['70% RH', '70% RH', '85% RH', '85% RH']

# Read data
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
            if 'fragments' in key or 'all' in key:
                mask = (0 < temp_PTR[key]['m153.061 (C[12]8H[1]9O[16]3) (Conc)']) & (temp_PTR[key]['m153.061 (C[12]8H[1]9O[16]3) (Conc)'] < 90)
                temp_PTR[key] = temp_PTR[key][mask]
            PTRMS[key] = temp_PTR[key]
    temp_AMS = import_data(f'{parent_path}{path}AMS/', '', 't_series', '%d-%m-%Y %H:%M:%S', 0)
    for key in temp_AMS.keys():
        if 'PToF' not in key:
            temp_AMS[key].columns = ['t_series', 'HROrg', 'HRNO3', 'HRSO4', 'HRNH4', 'HRChl', 'Ratio_H_C', 'Ratio_O_C', 
                            'familyCHN', 'familyCHO1', 'familyCHOgt1', 'familyCHO1N', 'familyCH', 'f43', 'f44', 'Time']
        AMS[key] = temp_AMS[key]
    temp_daq = import_data(f'{parent_path}{path}DAQ/', '', 'DAQ_Timestamp_UTC', '%d-%m-%Y %H:%M:%S', 0)
    for key in temp_daq.keys():
        DAQ[key] = temp_daq[key]

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)

# PTR-MS H:C and O:C calculation
bg_timestamps = [['2026-04-29 09:50', '2026-04-29 10:02'],
                 ['2026-04-30 07:29', '2026-04-30 07:36']]
for i, key in enumerate(['260429_VL+UV_RH85_all', '260430_VL+UV_RH85_all']):
    PTRMS[f'{key.split('_')[0]}_VL+UV_RH85_OC-HC'] = calc_OC_HC_PTRMS(PTRMS[key], bg_timestamps[i])

# Dataframe keys
SMPS_keys = [['260427_vanillin+UV_RH70_number', '260428_vanillin+UV_RH70_number', '260429_vanillin+UV_RH85_number', '260430_vanillin+UV_RH85_number'],
             ['260427_vanillin+UV_RH70_mass', '260428_vanillin+UV_RH70_mass', '260429_vanillin+UV_RH85_mass', '260430_vanillin+UV_RH85_mass']]
AMS_keys = ['260427_AMS_vanillin+UV_70RH_TS', '260428_AMS_vanillin+UV_70RH_TS', '260429_AMS_vanillin+UV_85RH_TS', '260430_AMS_vanillin+UV_85RH_TS']
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
ylim = [(50, 72), (38, 51)]

for i, key in enumerate(['260429_VL+UV_RH85_fragments', '260430_VL+UV_RH85_fragments']):
    PTR_df = time_filtered_conc(PTRMS[key], ['m153.061 (C[12]8H[1]9O[16]3) (Conc)'], timestamps[i+2])
    DAQ_df = time_filtered_conc(DAQ[DAQ_keys[i+2]], ['Temp_C'], timestamps[i+2])

    t_off = (pd.to_datetime(t_UV_off[i+2]) - pd.to_datetime(t_zero[i+2])) / pd.Timedelta(minutes = 1)

    fig, axes = plt.subplots(2, 1, figsize = (6.3, 6.5))
    axes[0].axvspan(0, t_off, color = 'y', alpha = 0.15, lw = 0, label = None)
    for ax in axes:
        plot_total(ax, PTR_df, 'm153.061 (C[12]8H[1]9O[16]3) (Conc)', 'indigo', t_zero[i+2])
        ax.tick_params(axis = 'y', labelcolor = 'indigo')
        ax.set_ylabel('C$_{8}$H$_{8}$O$_{3}$H$^{+}$ conc. (ppb)', color = 'indigo')
        ax.set(xlabel = 'Time (min)', title = f'{t_zero[i+2].split(' ')[0]}, 85% RH')

        ax_2 = ax.twinx()
        plot_total(ax_2, DAQ_df, 'Temp_C', 'tab:red', t_zero[i+2])
        ax_2.tick_params(axis = 'y', labelcolor = 'tab:red')
        ax_2.set_ylabel('Temperature ($^{\circ}$C)', color = 'tab:red')

    axes[1].set(xlim = (0, 100), ylim = ylim[i])

    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_VLvsTemp.jpg', dpi = 600)

#%%
# PTR-MS grouping of ions
for key in ['260429_VL+UV_RH85_products', '260430_VL+UV_RH85_products']:
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
#%%
# PTR-MS O:C and H:C
for i, key in enumerate(['260429_VL+UV_RH85_OC-HC', '260430_VL+UV_RH85_OC-HC']):
    fig, ax = vanKrevelen_ts(PTRMS[key], ['Ratio_H_C', 'Ratio_O_C'], None,
                             t_zero[i+2], [t_VL_inj[i], t_UV_off[i+2]], 5/60, f'{t_UV_off[i+2].split(' ')[0]}, 85% RH')
    
    fig.tight_layout(pad = 0.75)
    fig.savefig(f'{save_path}{timestamps[i+2][0].split(' ')[0]}_vanKrevelen_PTRMS.jpg', dpi = 600)
#%%
for i, key in enumerate(['260429_VL+UV_RH85_fragments', '260430_VL+UV_RH85_fragments']):
    fig, ax = plot_PTRMS_decay(PTRMS[key], 'm153.061 (C[12]8H[1]9O[16]3) (Conc)', list(PTRMS[key].keys()[:-2]), 
                               ['C$_{8}$H$_{8}$O$_{3}$H$^{+}$', 'C$_{5}$H$_{4}$H$^{+}$', 'C$_{6}$H$_{5}$O$_{2}$H$^{+}$', 
                                'C$_{6}$H$_{6}$O$_{2}$H$^{+}$', 'C$_{7}$H$_{8}$O$_{2}$H$^{+}$', 'C$_{8}$H$_{6}$O$_{3}$H$^{+}$'], 
                               t_zero[i+2], t_UV_off[i+2], timestamps[i+2][1], RH[i+2])
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_PTRMS_initial.jpg', dpi = 600)
#%%
# PTR-MS decay (wall loss corrected)
wall_loss = [0.0006446245895402325, 0.0004942765846080444]

for i, key in enumerate(['260429_VL+UV_RH85_fragments', '260430_VL+UV_RH85_fragments']):
    time_minutes = (PTRMS[key]['Time'] - pd.to_datetime(t_zero[i+2])) / pd.Timedelta(minutes = 1)
    to_replace = [t for t in time_minutes if t < 0]
    time_minutes = time_minutes.replace(to_replace, 0)

    PTRMS[key]['m153.061 (C[12]8H[1]9O[16]3) (Conc)'] = PTRMS[key]['m153.061 (C[12]8H[1]9O[16]3) (Conc)'] + time_minutes*wall_loss[i]*PTRMS[key]['m153.061 (C[12]8H[1]9O[16]3) (Conc)']
    
    fig, ax = plot_PTRMS_decay(PTRMS[key], 'm153.061 (C[12]8H[1]9O[16]3) (Conc)', list(PTRMS[key].keys()[:-2]), 
                               ['C$_{8}$H$_{8}$O$_{3}$H$^{+}$', 'C$_{5}$H$_{4}$H$^{+}$', 'C$_{6}$H$_{5}$O$_{2}$H$^{+}$', 
                                'C$_{6}$H$_{6}$O$_{2}$H$^{+}$', 'C$_{7}$H$_{8}$O$_{2}$H$^{+}$', 'C$_{8}$H$_{6}$O$_{3}$H$^{+}$'], 
                               t_zero[i+2], t_UV_off[i+2], timestamps[i+2][1], RH[i+2])
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_PTRMS_VLdecay_wall-loss-corrected.jpg', dpi = 600)
#%%
decays_minutes = np.array([0.0015498450850007783, 9.553880394080894e-05, 0.0015976754392935666, 5.9092508593650006e-05])
decays_seconds = decays_minutes / 60
print(decays_seconds)
#%%
# AMS and PTR-MS carbon oxidation state
COS_ylim = [(-1, 0), (-1.2, -0.2)]
for i, keys in enumerate([['260429_AMS_vanillin+UV_85RH_TS', '260429_VL+UV_RH85_OC-HC'],
                          ['260430_AMS_vanillin+UV_85RH_TS', '260430_VL+UV_RH85_OC-HC']]):

    AMS_bg = time_filtered_conc(AMS[keys[0]], ['HROrg'], HEPA_timestamps[i+2])
    AMS_DL = AMS_bg['HROrg'].std() * 3
    
    fig, axes, OC_ax = plot_COS(AMS[keys[0]], AMS_DL, PTRMS[keys[1]], t_VL_inj[i], t_zero[i+2], t_UV_off[i+2])

    OC_ax.set_ylim(0.3, 0.8)
    axes[1].set(ylim = COS_ylim[i])

    axes[0].set_title(f'{t_zero[i+2].split(' ')[0]}, 85% RH')

    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_COS.jpg', dpi = 600)