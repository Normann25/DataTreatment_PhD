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
paths = ['260505_Vanillin70ppb_UV_85RH_(NH4)2SO4-seeds/', '260506_Vanillin70ppb_UV_85RH_(NH4)2SO4-seeds/', 
         '260507_Vanillin70ppb_UV_dry_(NH4)2SO4-seeds/', '260508_Vanillin70ppb_UV_dry_(NH4)2SO4-seeds/']

timestamps = [['2026-05-05 07:40', '2026-05-05 17:59'],
              ['2026-05-06 07:12', '2026-05-06 16:23'],
              ['2026-05-07 11:06', '2026-05-07 18:42'],
              ['2026-05-08 07:59', '2026-05-08 16:20']]
t_injection = ['2026-05-05 12:27', '2026-05-06 10:45', '2026-05-07 13:06', '2026-05-08 10:49']
t_zero = ['2026-05-05 12:55', '2026-05-06 11:16', '2026-05-07 13:41', '2026-05-08 11:20']
t_UV_off = ['2026-05-05 16:59', '2026-05-06 15:23', '2026-05-07 17:42', '2026-05-08 15:20']
HEPA_timestamps = [['2026-05-05 07:47', '2026-05-05 08:07'],
                   ['2026-05-06 07:26', '2026-05-06 07:46'],
                   ['2026-05-07 08:10', '2026-05-07 08:30'],
                   ['2026-05-08 08:13', '2026-05-08 08:33']]

RH = ['85% RH', '85% RH', '0% RH', '0% RH']

SMPS = {}
# PTRMS = {}
# AMS = {}
# DAQ = {}
for t, path in zip(t_injection, paths):
    temp_SMPS = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp_SMPS.keys():
        temp_SMPS[key].loc[temp_SMPS[key]['Time'] < pd.to_datetime(t), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = 0
        # temp = remove_spikes_up(temp_SMPS[key], ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 20)
        # SMPS[key] = temp
        SMPS[key] = temp_SMPS[key]

save_path = '../../../Figures/Vanillin/2605_vanillin+UV+seeds/'

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)

SMPS_keys = [['260505_vanillin+UV+seeds_RH85_number', '260506_vanillin+UV+seeds_RH85_number', '260507_vanillin+UV+seeds_dry_number', '260508_vanillin+UV+seeds_dry_number'],
             ['260505_vanillin+UV+seeds_RH85_mass', '260506_vanillin+UV+seeds_RH85_mass', '260507_vanillin+UV+seeds_dry_mass', '260508_vanillin+UV+seeds_dry_mass']]
#%%
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260505_vanillin+UV+seeds_RH85_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, RH, 'Total concentration', t_zero, 2, 2, save_path)