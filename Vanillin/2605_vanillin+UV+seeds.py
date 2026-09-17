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

RH = ['85% RH', '85% RH', 'Dry', 'Dry']

SMPS = import_SMPS(paths, parent_path, 0)
# PTRMS = {}
AMS = import_AMS(paths, parent_path, 0)
DAQ = import_DAQ(paths, parent_path, 0)
for t, key in zip(t_injection, SMPS.keys()):
    SMPS[key].loc[SMPS[key]['Time'] < pd.to_datetime(t), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    SMPS[key] = remove_spikes_up(SMPS[key], ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 20)
for i, key in zip(t_injection, AMS.keys()):
    if 'PToF' not in key:
        AMS[key].loc[AMS[key]['Time'] < pd.to_datetime(t), ['Ratio_H_C', 'Ratio_O_C']] = np.nan
for key in DAQ.keys():
    DAQ[key] = remove_spikes(DAQ[key], ['Temp_C'], 5)

save_path = '../../../Figures/Vanillin/2605_vanillin+UV+seeds/'

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    # SMPS[key] = SMPS[key].fillna(0)

RH_mask = DAQ['DataDAQ_260507']['RH_Percent'] < 0.005
DAQ['DataDAQ_260507'] = DAQ['DataDAQ_260507'][RH_mask]

SMPS_keys = [['260505_vanillin+UV+seeds_RH85_number', '260506_vanillin+UV+seeds_RH85_number', '260507_vanillin+UV+seeds_dry_number', '260508_vanillin+UV+seeds_dry_number'],
             ['260505_vanillin+UV+seeds_RH85_mass', '260506_vanillin+UV+seeds_RH85_mass', '260507_vanillin+UV+seeds_dry_mass', '260508_vanillin+UV+seeds_dry_mass']]
AMS_keys = ['260505_AMS_vanillin+UV+seeds_85RH_TS', '260506_AMS_vanillin+UV+seeds_85RH_TS', '260507_AMS_vanillin+UV+seeds_dry_TS', '260508_AMS_vanillin+UV+seeds_dry_TS']
DAQ_keys = ['DataDAQ_260505', 'DataDAQ_260506', 'DataDAQ_260507', 'DataDAQ_260508']
#%%
for i, time in enumerate(timestamps):
    plot_AURA_overview(DAQ[DAQ_keys[i]], SMPS[SMPS_keys[0][i]], AMS[AMS_keys[i]], time, HEPA_timestamps[i], t_zero[i], RH[i], save_path)
#%%
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260505_vanillin+UV+seeds_RH85_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, RH, 'Total concentration', t_zero, 2, 2, save_path)
#%%
for i, key in enumerate(AMS_keys):
    plot_AMS(AMS[key], None, t_zero[i], timestamps[i], HEPA_timestamps[i], 1, RH[i], save_path)