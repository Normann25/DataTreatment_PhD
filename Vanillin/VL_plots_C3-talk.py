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
paths = ['260429_Vanillin70ppb_UV_RH85/', '260504_Vanillin70ppb_UV_dry/']
save_path = '../../../Figures/Vanillin/C3-retreat_talk/'

# Timestamps
timestamps = [['2026-04-29 08:52', '2026-04-29 16:45'],
              ['2026-05-04 08:45', '2026-05-04 15:11']]
t_humidify = ['2026-04-29 09:49']
t_inj = [['2026-04-29 10:02', '2026-04-29 11:18'], 
         ['2026-05-04 08:16', '2026-05-04 09:46']]
t_zero = ['2026-04-29 11:45', '2026-05-04 10:10']
t_off = ['2026-04-29 15:45', '2026-05-04 14:11']
HEPA_timestamps = [['2026-04-29 08:40', '2026-04-29 09:00'],
                   ['2026-05-04 08:10', '2026-05-04 08:30']]

# Dictionary keys
SMPS_keys = ['260429_vanillin+UV_RH85_number', '260504_vanillin+UV_dry_number']
AMS_keys = ['260429_AMS_vanillin+UV_85RH_TS', '260504_AMS_vanillin+UV_dry_TS']
PTRMS_keys = [['260429_VL+UV_RH85_fragments', 'm153.061 (C[12]8H[1]9O[16]3) (Conc)'],
              ['260504_VL+UV_dry_fragments', 'm153.060 (C[12]8H[1]9O[16]3) (Conc)']]

# Read data
SMPS = {}
SMPS_raw = import_SMPS(paths, parent_path, 0)
PTRMS = import_PTRMS(paths, parent_path)
AMS = import_AMS(paths, parent_path, 0)
for t, key in zip(t_zero, SMPS_keys):
    temp = SMPS_raw[key]
    temp.loc[temp['Time'] < pd.to_datetime(t), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    temp.loc[temp[temp.keys()[38]] <= 6, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    SMPS[key] = temp
for key in PTRMS.keys():
    if 'fragments' in key or 'all' in key:
        try:
            mask = (0 < PTRMS[key]['m153.060 (C[12]8H[1]9O[16]3) (Conc)']) & (PTRMS[key]['m153.060 (C[12]8H[1]9O[16]3) (Conc)'] < 90)
        except KeyError:
            mask = (0 < PTRMS[key]['m153.061 (C[12]8H[1]9O[16]3) (Conc)']) & (PTRMS[key]['m153.061 (C[12]8H[1]9O[16]3) (Conc)'] < 90)
    PTRMS[key] = PTRMS[key][mask]

wall_loss = [0.0006446245895402325, 0.001086]

for i, keys in enumerate(PTRMS_keys):
    time_minutes = (PTRMS[keys[0]]['Time'] - pd.to_datetime(t_zero[i])) / pd.Timedelta(minutes = 1)
    to_replace = [t for t in time_minutes if t < 0]
    time_minutes = time_minutes.replace(to_replace, 0)

    PTRMS[keys[0]][keys[1]] = PTRMS[keys[0]][keys[1]] + time_minutes*wall_loss[i]*PTRMS[keys[0]][keys[1]]
#%%
# PTR-MS decay
fig, ax = plt.subplots(figsize = (6.3, 3.5))
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0, 1, 4))
labels = ['Humid', 'Dry']

ax.axvspan(-175, -120, color = 'blue', alpha = 0.15, lw = 0, label = 'Humidification')
ax.axvspan(-105, -25, color = 'gray', alpha = 0.15, lw = 0, label = 'Injection')
ax.axvspan(0, 240, color = 'yellow', alpha = 0.15, lw = 0, label = 'UV on')

for i, key in enumerate(PTRMS_keys):
    temp = running_mean(PTRMS[key[0]], [key[1]], 'Time', '1min', None)
    time = (temp.index - pd.to_datetime(t_zero[i])) / pd.Timedelta(minutes = 1)

    ax.plot(time, temp[key[1]], color = colors[i*2], lw = 2, label = labels[i])

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles = handles[3:], labels = labels, fontsize = 12)

ax.tick_params(axis = 'both', labelsize = 12)
ax.set_ylabel('Concentration (ppb)', fontsize = 16)
ax.set_xlabel('Time (min)', fontsize = 16)
ax.set(ylim = (0, 95))

fig.tight_layout()
fig.savefig(f'{save_path}VL-decay.jpg', dpi = 600)
#%%
# PTR-MS concentration vs. m/z
PTR_products = [[113.059, 139.039, 141.055, 143.034, 169.049, 183.029],
                [141.055, 143.033, 169.049, 183.029]]

for i, key in enumerate(['260429_VL+UV_RH85_products', '260504_VL+UV_dry_products']):
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, 4))

    time = [pd.to_datetime(t_off[i]) - pd.Timedelta(minutes = 30), t_off[i]]

    concentration_cols = [col for col in PTRMS[key].columns if col.startswith('m') and '(' in col]

    temp = time_filtered_conc(PTRMS[key], concentration_cols, time)

    mz_values = []
    mean_concentrations = []
    for col in concentration_cols:
        matches = re.findall(r'-?\d*\.?\d+', col)
        mz_values.append(float(matches[0]))

        mean_concentrations.append(temp[col].mean())

    fig, ax = plt.subplots(figsize = (6.3, 3.5))

    ax.scatter(mz_values, mean_concentrations, color = colors[0], s = 50)
    ax.set(yscale = 'log', xlim = (40, 260))
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.set_ylabel('Concentration (ppb)', fontsize = 16)
    ax.set_xlabel('m/z', fontsize = 16)

    fig.tight_layout()
    fig.savefig(f'{save_path}{key.split('_')[2]}_PTR-ions.jpg', dpi = 600)

    filtered_data = [[],
                     []]
    for mz, conc in zip(mz_values, mean_concentrations):
        if mz in PTR_products[i]:
            filtered_data[0].append(mz)
            filtered_data[1].append(conc)

    fig2, ax2 = plt.subplots(figsize = (6.3, 3.5))

    ax2.scatter(mz_values, mean_concentrations, color = colors[0], s = 50)
    ax2.scatter(filtered_data[0], filtered_data[1], color = colors[2], s = 50)
    ax2.set(yscale = 'log', xlim = (40, 260))
    ax2.tick_params(axis = 'both', labelsize = 12)
    ax2.set_ylabel('Concentration (ppb)', fontsize = 16)
    ax2.set_xlabel('m/z', fontsize = 16)

    fig2.tight_layout()
    fig2.savefig(f'{save_path}{key.split('_')[2]}_PTR-ions_filtered.jpg', dpi = 600)
#%%
# SMPS
fig, ax = plt.subplots(figsize = (6.3, 3.5))
fig2, ax2 = plt.subplots(figsize = (6.3, 3.5))
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0, 1, 4))
labels = ['Humid', 'Dry']

ax.axvspan(0, 240, color = 'yellow', alpha = 0.15, lw = 0, label = 'UV on')
ax2.axvspan(0, 240, color = 'yellow', alpha = 0.15, lw = 0, label = 'UV on')

for i, key in enumerate(SMPS_keys):
    time = (SMPS[key]['Time'] - pd.to_datetime(t_zero[i])) / pd.Timedelta(minutes = 1)

    ax.plot(time, SMPS[key]['Geo. Mean (nm)'], color = colors[i*2], lw = 2, label = labels[i])

    ax2.plot(time, SMPS[key]['Total concentration'], color = colors[i*2], lw = 2, label = labels[i])

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles = handles[1:], labels = labels, fontsize = 12)

ax.tick_params(axis = 'both', labelsize = 12)
ax.set_ylabel('Geo. mean D$_{p}$ (nm)', fontsize = 16)
ax.set_xlabel('Time (min)', fontsize = 16)
ax.set(ylim = (0, 75), xlim = (-20, 300))

fig.tight_layout()
fig.savefig(f'{save_path}Geo_mean_Dp.jpg', dpi = 600)

handles, _ = ax2.get_legend_handles_labels()
ax2.legend(handles = handles[1:], labels = labels, fontsize = 12)

ax2.tick_params(axis = 'both', labelsize = 12)
ax2.set_ylabel('Number conc. (cm$^{-3}$)', fontsize = 16)
ax2.set_xlabel('Time (min)', fontsize = 16)
ax2.set(xlim = (-20, 300))

fig2.tight_layout()
fig2.savefig(f'{save_path}Total_conc_number.jpg', dpi = 600)
#%%
# AMS
fig, ax = plt.subplots(figsize = (6.3, 3.5))
fig2, ax2 = plt.subplots(figsize = (6.3, 3.5))
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0, 1, 4))
labels = ['Detection limit', 'Humid', 'Dry']
time_masks = [50, 100]

ax.axvspan(0, 240, color = 'yellow', alpha = 0.15, lw = 0, label = 'UV on')
ax2.axvspan(0, 240, color = 'yellow', alpha = 0.15, lw = 0, label = 'UV on')
ax.hlines(0.02, -20, 300, color = 'gray', ls = '--', lw = 1, label = 'Detection limit', zorder = 10)

for i, key in enumerate(AMS_keys):
    ams_bg = time_filtered_conc(AMS[key], ['HROrg'], HEPA_timestamps[i])

    OC_mask = AMS[key]['Time'] >= pd.to_datetime(t_zero[i]) + pd.Timedelta(minutes = time_masks[i])

    time = (AMS[key]['Time'] - pd.to_datetime(t_zero[i])) / pd.Timedelta(minutes = 1)

    ax.plot(time, AMS[key]['HROrg'], color = colors[i*2], lw = 2, label = labels[i])

    ax2.plot(time[OC_mask], AMS[key][OC_mask]['Ratio_O_C'], color = colors[i*2], lw = 2, label = labels[i])

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles = handles[1:], labels = labels, fontsize = 12)

ax.tick_params(axis = 'both', labelsize = 12)
ax.set_ylabel('SOA mass ($\mu$g m$^{-3}$)', fontsize = 16)
ax.set_xlabel('Time (min)', fontsize = 16)
ax.set(xlim = (-20, 300))

fig.tight_layout()
fig.savefig(f'{save_path}SOA_mass.jpg', dpi = 600)

handles, _ = ax2.get_legend_handles_labels()
ax2.legend(handles = handles[1:], labels = labels[1:], fontsize = 12)

ax2.tick_params(axis = 'both', labelsize = 12)
ax2.set_ylabel('O:C ratio', fontsize = 16)
ax2.set_xlabel('Time (min)', fontsize = 16)
ax2.set(xlim = (-20, 300), ylim = (-0.2, 2))

fig2.tight_layout()
fig2.savefig(f'{save_path}OC_ratio.jpg', dpi = 600)