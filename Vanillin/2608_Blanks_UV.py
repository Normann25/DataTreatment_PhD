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
paths = ['20260824_blank_UV/', '20260827_Blank_UV_85RH/', '20260828_Blank_UV/', '20260908_vanillin70ppb_UV_dry/', '20260909_vanillin70ppb_UV_85RH/']

timestamps = [['2026-08-24 12:45', '2026-08-24 18:21'],
              ['2026-08-27 08:20', '2026-08-27 13:45'],
              ['2026-08-28 08:40', '2026-08-28 13:55'],
              ['2026-09-08 10:30', '2026-09-08 17:20'],
              ['2026-09-09 09:50', '2026-09-09 16:30']]
t_zero = ['2026-08-24 13:20', '2026-08-27 08:40', '2026-08-28 08:55', '2026-09-08 12:20', '2026-09-09 11:30']
t_UV_off = ['2026-08-24 17:21', '2026-08-27 12:45', '2026-08-28 12:55', '2026-09-08 16:20', '2026-09-09 15:30']

RH = ['Dry', '85% RH', 'Dry', 'Dry', '85% RH']

SMPS = import_SMPS(paths, parent_path, 0)
PTRMS = import_PTRMS(paths, parent_path)
# AMS = {}
DAQ = import_DAQ(paths, parent_path, 0)
for key in DAQ.keys():
    DAQ[key] = remove_spikes(DAQ[key], ['Temp_C'], 5)

save_path = '../../../Figures/Vanillin/2608_Blanks/'

for key in SMPS.keys():
    if '85RH' in key and 'number' in key:
        SMPS[key].loc[SMPS[key]['Total concentration'] <= 10, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    if '260908' in key:
        SMPS[key].loc[SMPS[key]['Time'] < pd.to_datetime(t_zero[3]) + pd.Timedelta(minutes = 45), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    if 'vanillin' in key and 'mass' in key:
        SMPS[key].loc[SMPS[key]['Geo. Mean (nm)'] > 55, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan

RH_mask = DAQ['DataDAQ_260826']['RH_Percent'] > 10
DAQ['DataDAQ_260826'] = DAQ['DataDAQ_260826'][RH_mask]

ptr_mask = PTRMS['260908_VL+UV_dry_fragments']['m153.059 (C[12]8H[1]9O[16]3) (Conc)'] < 70
PTRMS['260908_VL+UV_dry_fragments'] = PTRMS['260908_VL+UV_dry_fragments'][ptr_mask]

SMPS_blank = [['20260824_blank_uv_dry_number', '20260827_blank_uv_85RH_number', '20260828_blank_uv_dry_number'],
             ['20260824_blank_uv_dry_mass', '20260827_blank_uv_85RH_mass', '20260828_blank_uv_dry_mass']]
SMPS_rep = [['260908_vanillin+UV_dry_number', '260909_vanillin+UV_85RH_number'],
            ['260908_vanillin+UV_dry_mass', '260909_vanillin+UV_85RH_mass']]
DAQ_keys = ['DataDAQ_260824', 'DataDAQ_260826', 'DataDAQ_260828', 'DataDAQ_260908', 'DataDAQ_260909']
PTRMS_keys = ['260824_blank_dry', '260827_blank_85RH', '260828_blank_dry', '260908_VL+UV_dry_fragments', '260909_VL+UV_85RH_fragments']
#%%
SMPS_number = SMPS_blank[0] + SMPS_rep[0]
for i, time in enumerate(timestamps):
    plot_AURA_overview(DAQ[DAQ_keys[i]], SMPS[SMPS_number[i]], None, time, None, t_zero[i], RH[i], save_path)
#%%
ax, ax_2 = plot_SMPS(SMPS, SMPS_rep, 'number and mass', timestamps[3:], 10, RH[3:], t_zero[3:], 1, 2, save_path)

ax, ax_2 = plot_SMPS(SMPS, SMPS_blank, 'number and mass', timestamps[:3], 10, RH[:3], t_zero[:3], 1, 3, save_path)
#%%
# PTR-MS decay (no wall loss correction)
for i, key in enumerate(PTRMS_keys[3:]):  
    fig, ax = plot_PTRMS_decay(PTRMS[key], 'm153.059 (C[12]8H[1]9O[16]3) (Conc)', None, ['C$_{8}$H$_{8}$O$_{3}$H$^{+}$'], 
                               t_zero[i+3], t_UV_off[i+3], timestamps[i+3][1], 'Dry')
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_PTRMS_initial.jpg', dpi = 600)
#%%
# PTR-MS decay (wall loss corrected)
wall_loss = [0.0013018008212910548, 0.0008360344956755417]

for i, key in enumerate(PTRMS_keys[3:]):
    time_minutes = (PTRMS[key]['Time'] - pd.to_datetime(t_zero[i+3])) / pd.Timedelta(minutes = 1)
    to_replace = [t for t in time_minutes if t < 0]
    time_minutes = time_minutes.replace(to_replace, 0)

    PTRMS[key]['m153.059 (C[12]8H[1]9O[16]3) (Conc)'] = PTRMS[key]['m153.059 (C[12]8H[1]9O[16]3) (Conc)'] + time_minutes*wall_loss[i]*PTRMS[key]['m153.059 (C[12]8H[1]9O[16]3) (Conc)']
    
    fig, ax = plot_PTRMS_decay(PTRMS[key], 'm153.059 (C[12]8H[1]9O[16]3) (Conc)', None, ['C$_{8}$H$_{8}$O$_{3}$H$^{+}$'], 
                               t_zero[i+3], t_UV_off[i+3], timestamps[i+3][1], 'Dry')
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_PTRMS_VLdecay_wall-loss-corrected.jpg', dpi = 600)
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