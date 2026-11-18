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
         ['2026-05-05 07:40', '2026-05-05 17:59'],
         ['2026-05-07 11:06', '2026-05-07 18:42']]
## Vanillin injection start
t_VL_inj = ['2026-05-01 08:40', '2026-04-29 10:02', '2026-05-05 10:45', '2026-05-07 11:17']
## UV on and off
t_UV_on = ['2026-05-01 10:38', '2026-04-29 11:45', '2026-05-05 12:55', '2026-05-07 13:41']
t_UV_off = ['2026-05-01 14:38', '2026-04-29 15:45', '2026-05-05 16:59', '2026-05-07 17:42']
## HEPA measurements on AMS
t_HEPA = [['2026-05-01 08:50', '2026-05-01 09:10'],
          ['2026-04-29 08:40', '2026-04-29 09:00'],
          ['2026-05-05 07:47', '2026-05-05 08:07'],
          ['2026-05-07 08:10', '2026-05-07 08:30']]

# Dataframe keys
SMPS_keys = [['260501_vanillin+UV_dry_number', '260429_vanillin+UV_RH85_number',
              '260505_vanillin+UV+seeds_RH85_number', '260507_vanillin+UV+seeds_dry_number'],
             ['260501_vanillin+UV_dry_mass', '260429_vanillin+UV_RH85_mass',
              '260505_vanillin+UV+seeds_RH85_mass', '260507_vanillin+UV+seeds_dry_mass']]
AMS_keys = ['260501_AMS_vanillin+UV_dry_TS', '260429_AMS_vanillin+UV_85RH_TS', 
            '260505_AMS_vanillin+UV+seeds_85RH_TS', '260507_AMS_vanillin+UV+seeds_dry_TS']
# PTRMS_keys = ['260501_VL+UV_dry_fragments', '260429_VL+UV_RH85_fragments', 
#               '', '']
DAQ_keys = ['DataDAQ_260501', 'DataDAQ_260429', 'DataDAQ_260505', 'DataDAQ_260507']

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
        temp.loc[temp['Total concentration'] <= 2, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        SMPS[row[i]] = remove_spikes_up(temp, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 20)
    # DAQ
    DAQ[key] = remove_spikes(DAQ[key], ['Temp_C'], 5)
    if 'dry' in SMPS_keys[0][i]:
        RH_mask = DAQ[key]['RH_Percent'] < 0.005
        DAQ[key] = DAQ[key][RH_mask]