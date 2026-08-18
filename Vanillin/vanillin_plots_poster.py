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
paths = ['260429_Vanillin70ppb_UV_RH85/', '260504_Vanillin70ppb_UV_dry/']
save_path = '../../../Figures/Vanillin/for_poster/'

# Timestamps
timestamps = [['2026-04-29 08:52', '2026-04-29 16:45'],
              ['2026-05-04 08:45', '2026-05-04 15:11']]
t_inj = [['2026-04-29 10:02', '2026-04-29 11:18'], 
         ['2026-05-04 10:10', '2026-05-04 09:46']]
t_zero = ['2026-04-29 11:45', '2026-05-04 10:10']
t_off = ['2026-04-29 15:45', '2026-05-04 14:11']
HEPA_timestamps = [['2026-04-29 08:40', '2026-04-29 09:00'],
                   ['2026-05-04 08:10', '2026-05-04 08:30']]

# Read data
SMPS = {}
SMPS_raw = {}
PTRMS = {}
AMS = {}
for t, path in zip(t_zero, paths):
    temp_smps = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp_smps.keys():
        SMPS_raw[key] = temp_smps[key]
        temp_smps[key].loc[temp_smps[key]['Time'] < pd.to_datetime(t), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        temp_smps[key].loc[temp_smps[key][temp_smps[key].keys()[38]] <= 6, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        temp = remove_spikes_up(temp_smps[key], ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 20)
        SMPS[key] = temp
    temp_PTR = import_PTRMS(f'{parent_path}{path}PTRMS/', '')
    for key in temp_PTR.keys():
        if 'fragments' in key or 'all' in key or 'filtered' in key:
            try:
                mask = (0 < temp_PTR[key]['m153.060 (C[12]8H[1]9O[16]3) (Conc)']) & (temp_PTR[key]['m153.060 (C[12]8H[1]9O[16]3) (Conc)'] < 90)
            except KeyError:
                mask = (0 < temp_PTR[key]['m153.061 (C[12]8H[1]9O[16]3) (Conc)']) & (temp_PTR[key]['m153.061 (C[12]8H[1]9O[16]3) (Conc)'] < 90)
            temp_PTR[key] = temp_PTR[key][mask]
        PTRMS[key] = temp_PTR[key]
    temp_AMS = import_data(f'{parent_path}{path}AMS/', '', 't_series', '%d-%m-%Y %H:%M:%S', 0)
    for key in temp_AMS.keys():
        if 'PToF' not in key:
            temp_AMS[key].columns = ['t_series', 'HROrg', 'HRNO3', 'HRSO4', 'HRNH4', 'HRChl', 'Ratio_H_C', 'Ratio_O_C', 
                            'familyCHN', 'familyCHO1', 'familyCHOgt1', 'familyCHO1N', 'familyCH', 'f43', 'f44', 'Time']
        AMS[key] = temp_AMS[key]

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)

SMPS_keys = ['260429_vanillin+UV_RH85_number', '260504_vanillin+UV_dry_number']
AMS_keys = ['260429_AMS_vanillin+UV_85RH_TS', '260504_AMS_vanillin+UV_dry_TS']
PTRMS_keys = [['260429_VL+UV_RH85_fragments', 'm153.061 (C[12]8H[1]9O[16]3) (Conc)'],
              ['260504_VL+UV_dry_fragments', 'm153.060 (C[12]8H[1]9O[16]3) (Conc)']]
wall_loss = [0.0006446245895402325, 0.001086]

for i, keys in enumerate(PTRMS_keys):
    time_minutes = (PTRMS[keys[0]]['Time'] - pd.to_datetime(t_zero[i])) / pd.Timedelta(minutes = 1)
    to_replace = [t for t in time_minutes if t < 0]
    time_minutes = time_minutes.replace(to_replace, 0)

    PTRMS[keys[0]][keys[1]] = PTRMS[keys[0]][keys[1]] + time_minutes*wall_loss[i]*PTRMS[keys[0]][keys[1]]
#%%
xlims = [(-180, 300), (-120, 300)]
ylims = [(-0.1, 1.60), (-0.02, 0.21)]
for i, key in enumerate(SMPS_keys):
    cmap = mpl.colormaps['viridis_r']
    colors = cmap(np.linspace(0, 1, 5))

    fig, axes = plt.subplots(2, 1, figsize = (7.5, 8.6), sharex = True)
    for ax in axes:
        ax.axvspan((pd.to_datetime(t_inj[i][0]) - pd.to_datetime(t_zero[i])) / pd.Timedelte(minutes = 1),
                   (pd.to_datetime(t_inj[i][1]) - pd.to_datetime(t_zero[i])) / pd.Timedelte(minutes = 1),
                   color = 'gray', alpha = 0.15, lw = 0)
        ax.axvspan(0, (pd.to_datetime(t_off[i]) - pd.to_datetime(t_zero[i])) / pd.Timedelte(minutes = 1),
                   color = 'yellow', alpha = 0.15, lw = 0)

    axes[0].plot((PTRMS[PTRMS_keys[i][0]]['Time'] - pd.to_datetime(t_zero[i])) / pd.Timedelta(minutes = 1), PTRMS[PTRMS_keys[i][0]][PTRMS_keys[i][1]],
               color = colors[1], lw = 2)
    axes[0].tick_params(axis = 'y', labelcolor = colors[1], labelsize = 18)
    axes[0].set_ylabel('VL conc. (ppb)', color = colors[1], fontsize = 24)
    axes[0].set_xlabel(None)

    ax0_twin = axes[0].twinx()
    ax0_twin.plot((AMS[AMS_keys[i]]['Time'] - pd.to_datetime(t_zero[i])) / pd.Timedelta(minutes = 1), AMS[AMS_keys[i]]['HROrg'],
                   color = colors[3], lw = 2)
    ax0_twin.tick_params(axis = 'y', labelcolor = colors[3], labelsize = 18)
    ax0_twin.set_ylabel('Org. mass ($\mu$g m$^{-3}$)', color = colors[3], fontsize = 24)
    ax0_twin.set(ylim = ylims[i])

    axes[1].plot((SMPS[key]['Time'] - pd.to_datetime(t_zero[i])) / pd.Timedelta(minutes = 1), SMPS[key]['Total concentration'], 
               color = colors[2], lw = 2)
    axes[1].tick_params(axis = 'y', labelcolor = colors[2], labelsize = 18)
    axes[1].set_ylabel('Number conc. (cm$^{-3}$)', color = colors[2], fontsize = 24)

    mask = SMPS[key]['Geo. Mean (nm)'] < 100
    temp = SMPS[key][mask]
    ax1_twin = axes[1].twinx()
    ax1_twin.plot((temp['Time'] - pd.to_datetime(t_zero[i])) / pd.Timedelta(minutes = 1), temp['Geo. Mean (nm)'], 
                  color = colors[4], lw = 2)
    ax1_twin.tick_params(axis = 'y', labelcolor = colors[4], labelsize = 18)
    ax1_twin.set_ylabel('Geo. mean D$_{p}$ (nm)', color = colors[4], fontsize = 24)

    axes[1].xaxis.set_major_locator(mpl.ticker.FixedLocator([-100, 0, 100, 200, 300]))
    axes[1].tick_params(axis = 'x', labelsize = 18)
    axes[1].set_xlabel('Time (min)', fontsize = 24)
    axes[1].set(xlim = xlims[i])

    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i].split(' ')[0]}_overview_ts.png', transparent = True, dpi = 600)
