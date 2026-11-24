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
path = '../../../Data/Processed data/2026_vanillin-campaign/'
save_path = '../../../Figures/Vanillin/'

# Timestamps
## Experiment duration
t_exp = [['2026-05-01 10:17', '2026-05-01 15:38'],
         ['2026-04-29 08:52', '2026-04-29 16:45'],
         ['2026-05-07 11:06', '2026-05-07 18:42'],
         ['2026-05-05 07:40', '2026-05-05 17:59']]
## Vanillin injection start
t_VL_inj = ['2026-05-01 08:40', '2026-04-29 10:02', '2026-05-07 11:17', '2026-05-05 10:45']
## UV on and off
t_UV_on = ['2026-05-01 10:38', '2026-04-29 11:45', '2026-05-07 13:41', '2026-05-05 12:55']
t_UV_off = ['2026-05-01 14:38', '2026-04-29 15:45', '2026-05-07 17:42', '2026-05-05 16:59']
## HEPA measurements on AMS
t_HEPA = [['2026-05-01 08:50', '2026-05-01 09:10'],
          ['2026-04-29 08:40', '2026-04-29 09:00'],
          ['2026-05-07 08:10', '2026-05-07 08:30'],
          ['2026-05-05 07:47', '2026-05-05 08:07']]

# Dataframe keys
SMPS_keys = [['260501_vanillin+UV_dry_number', '260429_vanillin+UV_RH85_number',
              '260507_vanillin+UV+seeds_dry_number', '260505_vanillin+UV+seeds_RH85_number'],
             ['260501_vanillin+UV_dry_mass', '260429_vanillin+UV_RH85_mass',
              '260507_vanillin+UV+seeds_dry_mass', '260505_vanillin+UV+seeds_RH85_mass']]
AMS_keys = ['260501_AMS_vanillin+UV_dry_TS', '260429_AMS_vanillin+UV_85RH_TS', 
            '260507_AMS_vanillin+UV+seeds_dry_TS', '260505_AMS_vanillin+UV+seeds_85RH_TS']
# PTRMS_keys = ['260501_VL+UV_dry_fragments', '260429_VL+UV_RH85_fragments', 
#               '', '']
DAQ_keys = ['DataDAQ_260501', 'DataDAQ_260429', 'DataDAQ_260507', 'DataDAQ_260505']

# Read data
SMPS_raw = import_SMPS([''], path, 0)
# PTRMS = import_PTRMS([''], path)
AMS = import_AMS([''], path, 0)
DAQ = import_DAQ([''], path, 0)

# Remove spikes from data
SMPS = {}
for i, key in enumerate(DAQ_keys):
    # SMPS
    for row in SMPS_keys:
        temp = SMPS_raw[row[i]]
        temp.loc[temp['Total concentration'] <= 100, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        SMPS[row[i]] = remove_spikes_up(temp, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 20)
    # DAQ
    DAQ[key] = remove_spikes(DAQ[key], ['Temp_C'], 5)
    if 'dry' in SMPS_keys[0][i]:
        RH_mask = DAQ[key]['RH_Percent'] < 0.005
        DAQ[key] = DAQ[key][RH_mask]

RH = ['dry', '85% RH', 'seeds, dry', 'seeds, 85% RH']

def AURA_overview_figure(daq, daq_keys, smps, smps_keys, ams, ams_keys, timestamps, bg_timestamps, t_zero, nrows, ncols, RH, Dp_ylim, save_path):
    def plot_AURA(ax, twin_ax, daq, smps, ams, timestamps, bg_timestamps, t_zero, RH):
        if ams is not None:
            ams_bg = time_filtered_conc(ams, ['HROrg'], bg_timestamps)
            Org_DL = ams_bg['HROrg'].std() * 3
            ams.loc[ams['HROrg'] < Org_DL, ['Ratio_H_C', 'Ratio_O_C']] = np.nan
            ams = time_filtered_conc(ams, ['Ratio_H_C', 'Ratio_O_C'], timestamps)

            plot_total(ax[3], ams, 'Ratio_O_C', 'tab:cyan', t_zero)

        daq = time_filtered_conc(daq, ['Temp_C', 'RH_Percent', 'Laser_Distance'], timestamps)
        smps = time_filtered_conc(smps, ['Geo. Mean (nm)', 'Total concentration'], timestamps)

        plot_total(ax[0], daq, 'Temp_C', 'tab:red', t_zero)
        ax0_2 = twin_ax[0]
        plot_total(ax0_2, daq, 'RH_Percent', 'tab:blue', t_zero)

        plot_total(ax[1], daq, 'Laser_Distance', 'darkred', t_zero)

        plot_total(ax[2], smps, 'Total concentration', 'purple', t_zero)
        ax2_2 = twin_ax[1]
        plot_total(ax2_2, smps, 'Geo. Mean (nm)', 'green', t_zero)

        ax[0].set_title(f'{t_zero.split(' ')[0]} \n {RH}')

        return

    def set_ax_titles(ax, twin_ax, nrows):

        ax[0].set_ylabel('T ($^{\circ}$C)', color = 'tab:red')
        twin_ax[0].set_ylabel('RH (%)', color = 'tab:blue')

        ax[1].set_ylabel('LD (cm)', color = 'darkred')

        ax[2].set_ylabel('N (# cm$^{-3}$)', color = 'purple')
        twin_ax[1].set_ylabel('D$_{p}$ (nm)', color = 'green')

        if nrows >= 4:
            ax[3].set_ylabel('O:C', color = 'tab:cyan')

        return

    def set_ax_labelcolors(ax, twin_ax, nrows):

        ax[0].tick_params(axis = 'y', labelcolor = 'tab:red', color = 'tab:red', which = 'both')
        ax[0].spines['left'].set_color('tab:red')
        ax[0].spines['right'].set_color('tab:blue')
        ax[0].spines['left'].set_linewidth(0.9)
        ax[0].spines['right'].set_linewidth(0.9)
        ax[0].set_xlabel(None)
        twin_ax[0].tick_params(axis = 'y', labelcolor = 'tab:blue', color = 'tab:blue', which = 'both')
        twin_ax[0].spines['left'].set_color('tab:red')
        twin_ax[0].spines['right'].set_color('tab:blue')

        ax[1].tick_params(axis = 'y', labelcolor = 'darkred', color = 'darkred', which = 'both')
        ax[1].spines['left'].set_color('darkred')
        ax[1].spines['left'].set_linewidth(0.9)
        ax[1].set_xlabel(None)

        ax[2].tick_params(axis = 'y', labelcolor = 'purple', color = 'purple', which = 'both')
        ax[2].spines['left'].set_color('purple')
        ax[2].spines['right'].set_color('green')
        ax[2].spines['left'].set_linewidth(0.9)
        ax[2].spines['right'].set_linewidth(0.9)
        twin_ax[1].tick_params(axis = 'y', labelcolor = 'green', color = 'green', which = 'both')
        twin_ax[1].spines['left'].set_color('purple')
        twin_ax[1].spines['right'].set_color('green')

        if nrows >= 4:
            ax[2].set_xlabel(None)
            ax[3].tick_params(axis = 'y', labelcolor = 'tab:cyan', color = 'tab:cyan', which = 'both')
            ax[3].spines['left'].set_color('tab:cyan')
            ax[3].spines['left'].set_linewidth(0.9)

        return

    fig, ax = plt.subplots(nrows, ncols, figsize = (ncols*2.5, nrows*2), sharex = 'col', sharey = 'row')

    if ncols > 1:
        twin_ax = []
        ax = ax.T
        for i in range(ncols):
            if ams is not None:
                ams_df, bg_time = ams[ams_keys[i]], bg_timestamps[i]
            else:
                ams_df, bg_time = None, None

            x_twins = [ax[i][0].twinx(), ax[i][2].twinx()]
            twin_ax.append(x_twins)

            plot_AURA(ax[i], x_twins, daq[daq_keys[i]], smps[smps_keys[i]], ams_df, 
                      timestamps[i], bg_time, t_zero[i], RH[i])
            set_ax_labelcolors(ax[i], x_twins, nrows)

        twin_ax[-1][1].set(ylim = Dp_ylim)

        for row in np.array(twin_ax).T:
            ref_ax = row[-1]
            for x_twin in row[:-1]:
                x_twin.sharey(ref_ax)
                x_twin.tick_params(axis = 'y', which = 'both', labelright = False)

        set_ax_titles(ax[0], twin_ax[-1], nrows)
        date = f'{t_zero[0].split(' ')[0]}_{t_zero[-1].split(' ')[0]}'
    else:
        twin_ax = ax[0].twinx(), ax[2].twinx()
        plot_AURA(ax, twin_ax, daq, smps, ams, timestamps, bg_timestamps, t_zero, RH)
        set_ax_labelcolors(ax, twin_ax, nrows)
        set_ax_titles(ax, twin_ax, nrows)
        date = t_zero[0].split(' ')[0]

    fig.tight_layout()
    fig.savefig(f'{save_path}{date}_AURA_overview.jpg', dpi = 600)

    return fig, ax

fig, ax = AURA_overview_figure(DAQ, DAQ_keys, SMPS, SMPS_keys[0], AMS, AMS_keys, 
                               t_exp, t_HEPA, t_UV_on, 4, 4, RH, (5, 100), save_path)