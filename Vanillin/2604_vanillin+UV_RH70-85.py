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
t_zero = ['2026-04-27 12:16', '2026-04-28 12:35', '2026-04-29 11:45', '2026-04-30 12:47']
t_UV_off = ['2026-04-28 16:36', '2026-04-29 15:45', '2026-04-30 16:47']
RH = ['70% RH', '70% RH', '85% RH', '85% RH']

SMPS = {}
PTRMS = {}
DAQ = {}
for t, path in zip(t_zero, paths):
    temp_SMPS = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp_SMPS.keys():
        temp_SMPS[key].loc[temp_SMPS[key]['Time'] < pd.to_datetime(t), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        SMPS[key] = temp_SMPS[key]
    if path != paths[0]:
        temp_PTR = import_PTRMS(f'{parent_path}{path}PTRMS/', '')
        for key in temp_PTR.keys():
            mask = (0 < temp_PTR[key][temp_PTR[key].keys()[5]]) & (temp_PTR[key][temp_PTR[key].keys()[5]] < 150)
            temp = temp_PTR[key][mask]
            PTRMS[key] = temp
    temp_daq = import_data(f'{parent_path}{path}DAQ/', '', 'DAQ_Timestamp_UTC', '%d-%m-%Y %H:%M:%S', 0)
    for key in temp_daq.keys():
        DAQ[key] = temp_daq[key]

save_path = 'Figures/2604_vanillin+UV_RH70-85/'

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)
#%%
SMPS_keys = [['260427_vanillin+UV_RH70_number', '260428_vanillin+UV_RH70_number', '260429_vanillin+UV_RH85_number', '260430_vanillin+UV_RH85_number'],
             ['260427_vanillin+UV_RH70_mass', '260428_vanillin+UV_RH70_mass', '260429_vanillin+UV_RH85_mass', '260430_vanillin+UV_RH85_mass']]

ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260428_vanillin+UV_RH70_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, RH, 'Total concentration', t_zero, 2, 2, save_path)
#%%
PTRMS_keys = ['260428_VL+UV_70RH_initial', '260429_VL+UV_RH85_initial', '260430_VL+UV_RH85_inital']
for i, key in enumerate(PTRMS_keys):
    fig, ax = plot_PTRMS_decay(PTRMS[key], PTRMS[key].keys()[5], [PTRMS[key].keys()[4]], 
                               ['C$_{8}$H$_{9}$O$_{3}^{+}$', 'C$_{7}$H$_{9}$O$_{2}^{+}$'], 
                               t_zero[i+1], t_UV_off[i], timestamps[i+1][1], RH[i+1])
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+1].split(' ')[0]}_PTRMS_initial.jpg', dpi = 600)
