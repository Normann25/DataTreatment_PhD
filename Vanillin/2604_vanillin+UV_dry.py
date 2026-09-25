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
# Paths
parent_path = '../../../Data/2026/'
paths = ['260421_Vanillin70ppb_UV_dry_v2/', '260422_Vanillin70ppb_UV_dry/', '260501_Vanillin70ppb_UV_dry/', '260504_Vanillin70ppb_UV_dry/']
save_path = '../../../Figures/Vanillin/2604_vanillin+UV_dry/'

# Timestamps
timestamps = [['2026-04-21 09:11', '2026-04-21 17:01'],
              ['2026-04-22 08:50', '2026-04-22 16:14'],
              ['2026-05-01 10:17', '2026-05-01 15:38'],
              ['2026-05-04 08:45', '2026-05-04 15:11']]
t_VL_inj = ['2026-05-01 08:40', '2026-05-04 08:16']
t_zero = ['2026-04-21 11:28', '2026-04-22 10:42', '2026-05-01 10:38', '2026-05-04 10:10']
t_UV_off = ['2026-05-01 14:38', '2026-05-04 14:11']
HEPA_timestamps = [['2026-04-21 08:40', '2026-04-21 09:00'],
                   ['2026-04-22 08:25', '2026-04-22 08:40'],
                   ['2026-05-01 08:50', '2026-05-01 09:10'],
                   ['2026-05-04 08:10', '2026-05-04 08:30']]

# Dataframe keys
SMPS_keys = [['260421_vanillin+UV_dry_number', '260422_vanillin+UV_dry_number', '260501_vanillin+UV_dry_number', '260504_vanillin+UV_dry_number'], 
             ['260421_vanillin+UV_dry_mass', '260422_vanillin+UV_dry_mass', '260501_vanillin+UV_dry_mass', '260504_vanillin+UV_dry_mass']]
AMS_keys = ['260421_AMS_vanillin+UV_dry_TS', '260422_AMS_vanillin+UV_dry_TS', '260501_AMS_vanillin+UV_dry_TS', '260504_AMS_vanillin+UV_dry_TS']
DAQ_keys = ['DataDAQ_260421', 'DataDAQ_260422', 'DataDAQ_260501', 'DataDAQ_260504']
PTRMS_keys = ['260501_VL+UV_dry_fragments', '260504_VL+UV_dry_fragments', '260501_VL+UV_dry_all', '260504_VL+UV_dry_all']

# Read data
SMPS = {}
SMPS_raw = import_SMPS(paths, parent_path, 0)
PTRMS = import_PTRMS(paths, parent_path)
AMS = import_AMS(paths, parent_path, 0)
DAQ = import_DAQ(paths, parent_path, 0)
for i, key in enumerate(np.array(SMPS_keys).flatten()):
    if 'mass' in key:
        t = t_zero[i-4]
        temp = remove_spikes_up(SMPS_raw[key], [SMPS_raw[key].keys()[38]], max(SMPS_raw[key][SMPS_raw[key].keys()[38]])/4)
    else:
        t = t_zero[i]
        temp = SMPS_raw[key]
    temp.loc[temp['Time'] < pd.to_datetime(t) + pd.Timedelta(minutes = 30), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    temp.loc[temp[SMPS_raw[key].keys()[38]] == 0, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    temp = remove_spikes_up(temp, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 50)
    temp = remove_spikes_down(temp, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 50)
    temp.rename(columns = {temp.columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = temp
for i, key in enumerate(PTRMS_keys):
    if 'fragments' in key:
        mask = (0 < PTRMS[key]['m153.060 (C[12]8H[1]9O[16]3) (Conc)']) & (PTRMS[key]['m153.060 (C[12]8H[1]9O[16]3) (Conc)'] < 90)
        temp = PTRMS[key][mask]
        temp = wall_loss_corr(temp, ['m153.060 (C[12]8H[1]9O[16]3) (Conc)'], t_zero[i+2], 
                              [pd.to_datetime(t_UV_off[i]) + pd.Timedelta(minutes = 10), timestamps[i+2][1]])
    if 'all' in key:
        mask = (0 < PTRMS[key]['m153.060 (C[12]8H[1]9O[16]3) (Conc)']) & (PTRMS[key]['m153.060 (C[12]8H[1]9O[16]3) (Conc)'] < 90)
        temp = PTRMS[key][mask]
        temp = wall_loss_corr(temp, ['m153.060 (C[12]8H[1]9O[16]3) (Conc)'], t_zero[i], 
                              [pd.to_datetime(t_UV_off[i-2]) + pd.Timedelta(minutes = 10), timestamps[i][1]])
    PTRMS[key] = temp
for key in DAQ.keys():
    DAQ[key] = remove_spikes(DAQ[key], ['Temp_C', 'RH_Percent'], 5)

# PTR-MS H:C and O:C calculation
PTR_merge_keys = [['260501_VL+UV_dry_fragments', '260501_VL+UV_dry_products'],
                  ['260504_VL+UV_dry_fragments', '260504_VL+UV_dry_products']]
bg_timestamps = [['2026-05-01 08:00', '2026-05-01 08:40'],
                 ['2026-05-04 08:01', '2026-05-04 08:16']]

for i, key in enumerate(['260501_VL+UV_dry_all', '260504_VL+UV_dry_all']):
    PTRMS[f'{key.split('_')[0]}_VL+UV_dry_OC-HC'] = calc_OC_HC_PTRMS(PTRMS[key], bg_timestamps[i])
    PTRMS[f'{key.split('_')[0]}_VL+UV_dry_OC-HC_filtered'] = calc_OC_HC_PTRMS(PTRMS[f'{key.split('_')[0]}_VL+UV_dry_filtered'], bg_timestamps[i])
#%%
# Experiment overview
for i, time in enumerate(timestamps):
    plot_AURA_overview(DAQ[DAQ_keys[i]], SMPS[SMPS_keys[0][i]], AMS[AMS_keys[i]], time, HEPA_timestamps[i], t_zero[i], 'Dry', save_path)
#%%
# SMPS raw
for i, time in enumerate(timestamps):
    df_number = time_filtered_conc(SMPS_raw[SMPS_keys[0][i]], [SMPS_raw[SMPS_keys[0][i]].keys()[38], 'Geo. Mean (nm)'], time)
    df_mass = time_filtered_conc(SMPS_raw[SMPS_keys[1][i]], [SMPS_raw[SMPS_keys[1][i]].keys()[38], 'Geo. Mean (nm)'], time)
    print(df_mass)
    fig, ax = plt.subplots(2, 1, figsize = (6.3, 6.3))

    plot_total(ax[0], df_number, SMPS_raw[SMPS_keys[0][i]].keys()[38], 'purple', t_zero[i])
    ax[0].set_ylabel('Total number conc. (# cm$^{-3}$)', color = 'purple')
    ax[0].tick_params(axis = 'y', labelcolor = 'purple')
    ax0_twin = ax[0].twinx()
    plot_total(ax0_twin, df_number, 'Geo. Mean (nm)', 'green', t_zero[i])
    ax0_twin.set_ylabel('Geo. mean D$_{p}$ (nm)', color = 'green')
    ax0_twin.tick_params(axis = 'y', labelcolor = 'green')

    plot_total(ax[1], df_mass, SMPS_raw[SMPS_keys[1][i]].keys()[38], 'purple', t_zero[i])
    ax[1].set_ylabel('Total mass conc. ($\mu$g m$^{-3}$)', color = 'purple')
    ax[1].tick_params(axis = 'y', labelcolor = 'purple')
    ax1_twin = ax[1].twinx()
    plot_total(ax1_twin, df_mass, 'Geo. Mean (nm)', 'green', t_zero[i])
    ax1_twin.set_ylabel('Geo. mean D$_{p}$ (nm)', color = 'green')
    ax1_twin.tick_params(axis = 'y', labelcolor = 'green')

    fig.suptitle(f'{t_zero[i].split(' ')[0]}, Dry', fontsize = 14)
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i].split(' ')[0]}_SMPS_TS_raw.jpg', dpi = 600)
#%%
# SMPS spikes removed
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, 'number and mass', timestamps, 10, ['Dry']*len(t_zero), t_zero, 2, 2, save_path)
#%%
# AMS
for i, key in enumerate(AMS_keys):
    plot_AMS(AMS[key], None, t_zero[i], timestamps[i], HEPA_timestamps[i], 1, 'Dry', save_path)
#%%
# AMS pie charts
for time, key in zip(t_zero[2:], AMS_keys[2:]):
    temp = AMS[key]
    temp['Time (min)'] = (temp['Time'].dt.floor('min') - pd.to_datetime(time)) / pd.Timedelta(minutes = 1)
    get_times = np.array([[118, 122],
                          [178, 182],
                          [238, 242]])

    species = ['familyCHO1', 'familyCHOgt1', 'familyCH']
    cmap = mpl.colormaps['viridis_r']
    colors = cmap(np.linspace(0, 1, len(species)+2))

    piechart_values = np.zeros((3, 3))
    for i, get_t in enumerate(get_times):
        filtered_temp = temp.loc[(temp['Time (min)'] >= get_t[0]) & (temp['Time (min)'] <= get_t[1])]
        filtered_temp['sum'] = filtered_temp[species].sum(axis = 1)
        for j, specie in enumerate(species):
            piechart_values[i][j] += (filtered_temp[specie].mean() / filtered_temp['sum'].mean()) * 100

        fig, ax = plt.subplots(figsize = (5, 5))
        ax.pie(piechart_values[i], labels = species, colors = colors, autopct = '%1.2f%%')
        ax.set(title = f'{time.split(' ')[0]}, 85% RH, {get_t[0]+2} min')

        fig.tight_layout()
        fig.savefig(f'{save_path}{time.split(' ')[0]}_AMSpie_{get_t[0]+2}min.jpg', dpi = 600)
#%%
# PTR-MS VL concentration and chamber temperature
ylim = [(45, 65), (60, 90)]

for i, key in enumerate(PTRMS_keys[:2]):
    PTR_df = time_filtered_conc(PTRMS[key], ['m153.060 (C[12]8H[1]9O[16]3) (Conc)'], timestamps[i+2])
    DAQ_df = time_filtered_conc(DAQ[DAQ_keys[i+2]], ['Temp_C'], timestamps[i+2])

    t_off = (pd.to_datetime(t_UV_off[i]) - pd.to_datetime(t_zero[i+2])) / pd.Timedelta(minutes = 1)

    fig, axes = plt.subplots(2, 1, figsize = (6.3, 6.5))
    axes[0].axvspan(0, t_off, color = 'y', alpha = 0.15, lw = 0, label = None)
    for ax in axes:
        plot_total(ax, PTR_df, 'm153.060 (C[12]8H[1]9O[16]3) (Conc)', 'indigo', t_zero[i+2])
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
for key in ['260501_VL+UV_dry_products', '260504_VL+UV_dry_products']:
    # Identify concentration columns
    concentration_cols = [col for col in PTRMS[key].columns if col.startswith('m') and '(' in col] # The name of the time series
    smooth_data_array = GetData(PTRMS[key], concentration_cols, smooth=True, window_size=12, normalize = True)
    data_array = GetData(PTRMS[key], concentration_cols, smooth=False, window_size=50, normalize = True)

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
# PTR-MS final ion groups
PTR_cluster_number = [1, 4, 3, 6, 1, 1, 1, 6, 4, 9, 6, 10, 1, 9, 2, 4, 7, 3, 10, 3, 4, 5, 2, 10, 5, 7, 
                      4, 7, 3, 5, -1, 4, -1, 8, 6, -1, 5, -1, 6, 10, 8, -1, 9, 10, -1, 10, 11, 11, 11, -1]

for i, keys in enumerate(PTR_merge_keys):
    merged = pd.merge(PTRMS[keys[0]], PTRMS[keys[1]], on = 'Time', how = 'outer')
    mask = (0 < merged['m153.060 (C[12]8H[1]9O[16]3) (Conc)']) & (merged['m153.060 (C[12]8H[1]9O[16]3) (Conc)'] < 90)
    merged = merged[mask]

    concentration_cols = [col for col in PTRMS[keys[1]].columns if col.startswith('m') and '(' in col]
    concentration_cols = np.array([PTR_cluster_number, concentration_cols]).T
    unique_clusters = set(map(lambda x:int(x[0]), concentration_cols))
    grouped_concentration_cols = [[y[1] for y in concentration_cols if int(y[0])==x] for x in unique_clusters]

    VL_cols = [col for col in PTRMS[keys[0]].columns if col.startswith('m') and '(' in col]

    fig, axes = plt.subplots(13, 1, figsize = (8, 3.1*13), sharex = True)

    cmap = mpl.colormaps['viridis']
    colors_VL = cmap(np.linspace(0, 1, len(VL_cols)+1))

    for j, col in enumerate(VL_cols):
        plot_total(axes[0], merged, col, colors_VL[j], t_zero[i+2])
    axes[0].legend(labels = VL_cols)
    axes[0].set(xlabel = None, ylabel = 'Conc. (ppb)', title = 'VL fragments')

    for k, cluster in enumerate(list(unique_clusters)):
        colors = cmap(np.linspace(0, 1, len(grouped_concentration_cols[k])+1))
        for l, col in enumerate(grouped_concentration_cols[k]):
            plot_total(axes[cluster], merged, col, colors[l], t_zero[i+2])

        axes[cluster].legend(labels = grouped_concentration_cols[k])
        axes[cluster].set(xlabel = None, ylabel = 'Conc. (ppb)', title = f'Cluster {cluster}')

    axes[-1].set(xlabel = 'Time')
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_PTR-MS_clusters.jpg', dpi = 600)
#%%
PTR_filtered_clusters = [0, 2, 5, 5, 3, 1, 1, 3, 3, 0, 1, -1, 0, 0, 0, -1, 2, 1, -1, 2, 0, 0, 4, -1, 5, -1]

for i, key in enumerate(['260501_VL+UV_dry_filtered', '260504_VL+UV_dry_filtered']):
    concentration_cols = [col for col in PTRMS[key].columns if col.startswith('m') and '(' in col]
    concentration_cols = np.array([PTR_filtered_clusters, concentration_cols]).T
    unique_clusters = set(map(lambda x:int(x[0]), concentration_cols))
    grouped_concentration_cols = [[y[1] for y in concentration_cols if int(y[0])==x] for x in unique_clusters]

    fig, axes = plt.subplots(7, 1, figsize = (8, 3.1*7), sharex = True)

    cmap = mpl.colormaps['viridis']
    colors_VL = cmap(np.linspace(0, 1, 8))

    for k, cluster in enumerate(list(unique_clusters)):
        colors = cmap(np.linspace(0, 1, len(grouped_concentration_cols[k])+1))
        for l, col in enumerate(grouped_concentration_cols[k]):
            plot_total(axes[cluster], PTRMS[key], col, colors[l], t_zero[i+2])

        axes[cluster].legend(labels = grouped_concentration_cols[k])
        axes[cluster].set(xlabel = None, ylabel = 'Conc. (ppb)', title = f'Cluster {cluster}')

    axes[-1].set(xlabel = 'Time')
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_PTR-MS_clusters.jpg', dpi = 600)
#%%
# PTR-MS O:C and H:C
for i, key in enumerate(['260501_VL+UV_dry_OC-HC', '260504_VL+UV_dry_OC-HC']):
    fig, ax = vanKrevelen_ts(PTRMS[key], ['Ratio_H_C', 'Ratio_O_C'], None,
                             t_zero[i+2], [timestamps[i+2][0], t_UV_off[i]], 5/60, f'{timestamps[i+2][0].split(' ')[0]}, Dry')
    fig.tight_layout(pad = 0.75)
    fig.savefig(f'{save_path}{timestamps[i+2][0].split(' ')[0]}_vanKrevelen_PTRMS.jpg', dpi = 600)

    fig2, ax2 = vanKrevelen_ts(PTRMS[f'{key}_filtered'], ['Ratio_H_C', 'Ratio_O_C'], None,
                                 t_zero[i+2], [timestamps[i+2][0], t_UV_off[i]], 5/60, f'{timestamps[i+2][0].split(' ')[0]}, Dry (filtered)')
    ax2[1].set(ylim = (1.126, 1.1325), xlim = (0.348, 0.36))
    fig2.tight_layout(pad = 0.75)
    fig2.savefig(f'{save_path}{timestamps[i+2][0].split(' ')[0]}_vanKrevelen_PTRMS_filtered.jpg', dpi = 600)
#%%
# PTR-MS decay (wall loss corrected)
for i, key in enumerate(PTRMS_keys[:2]):  
    fig, ax = plot_PTRMS_decay(PTRMS[key], 'm153.060 (C[12]8H[1]9O[16]3) (Conc)', list(PTRMS[key].keys()[:-2]), 
                               ['C$_{8}$H$_{8}$O$_{3}$H$^{+}$', 'C$_{5}$H$_{4}$H$^{+}$', 'C$_{7}$H$_{6}$O$_{2}$H$^{+}$', 
                                'C$_{7}$H$_{8}$O$_{2}$H$^{+}$', 'C$_{7}$H$_{5}$O$_{3}$H$^{+}$', 'C$_{8}$H$_{6}$O$_{3}$H$^{+}$'], 
                               t_zero[i+2], t_UV_off[i], timestamps[i+2][1], 'Dry')
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_PTRMS_initial.jpg', dpi = 600)
#%%
decays_minutes = np.array([0.00123, 0.000258, 0.00102, 0.000223])
errors_minutes = np.array([0.00103, 0.00980, 0.00105, 0.00980])

decays_seconds = decays_minutes / 60
errors_seconds = errors_minutes / 60
print(decays_seconds)
print(errors_seconds)

from sympy import *
# Define variables
k1, k2, k_mean = symbols('k1, k2, k_mean')
dk1, dk2, dk_mean = symbols('sigma_k1, sigma_k2, sigma_k_mean')

# Define relation
k_mean = (k1 + k2) / 2

# Calculate uncertainty
dk_mean = sqrt((k_mean.diff(k1) * dk1)**2 + (k_mean.diff(k2) * dk2)**2)

# Turn expressions into numerical functions
fk_mean = lambdify((k1, k2), k_mean)
fdk_mean = lambdify((k1, dk1, k2, dk2), dk_mean)

for i in range(2):
    # Define values and their errors
    vk1, vdk1 = decays_seconds[i], errors_seconds[i]
    vk2, vdk2 = decays_seconds[i+2], errors_seconds[i+2]

    # Numerically evaluate expressions
    vk_mean = fk_mean(vk1, vk2)
    vdk_mean = fdk_mean(vk1, vdk1, vk2, vdk2)
    print(f'k_mean = {vk_mean} +- {vdk_mean}')
#%%
# AMS and PTR-MS carbon oxidation state
for i, keys in enumerate([['260501_AMS_vanillin+UV_dry_TS', '260501_VL+UV_dry_OC-HC'],
                          ['260504_AMS_vanillin+UV_dry_TS', '260504_VL+UV_dry_OC-HC']]):

    AMS_bg = time_filtered_conc(AMS[keys[0]], ['HROrg'], HEPA_timestamps[i+2])
    AMS_DL = AMS_bg['HROrg'].std() * 3
    
    fig, axes, OC_ax = plot_COS(AMS[keys[0]], AMS_DL, PTRMS[keys[1]], t_VL_inj[i], t_zero[i+2], t_UV_off[i])

    axes[0].set_title(f'{t_zero[i+2].split(' ')[0]}, Dry')

    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+2].split(' ')[0]}_COS.jpg', dpi = 600)