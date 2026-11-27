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
parent_path = '../../../Data/Processed data/2026_vanillin-campaign/'
save_path = '../../../Figures/Vanillin/'
Fig_folders = ['2604_vanillin+UV_dry']*2 + ['2604_vanillin+UV_RH70-85']*4 + ['2604_vanillin+UV_dry']*2 + ['2605_vanillin+UV+seeds']*4

# Timestamps
timestamps = [
            #   ['2026-01-13 14:36', '2026-01-13 18:25'],
#               ['2026-01-15 11:10', '2026-01-15 17:10'],
#               ['2026-01-16 12:44', '2026-01-16 17:56'],
              ['2026-04-21 09:11', '2026-04-21 11:28', '2026-04-21 17:01'],
              ['2026-04-22 08:50', '2026-04-22 10:42', '2026-04-22 16:14'],
              ['2026-04-23 08:50', '2026-04-23 11:58', '2026-04-23 15:59'],
            #   ['2026-04-27 12:16', '2026-04-27 16:21'],
              ['2026-04-28 11:07', '2026-04-28 12:35', '2026-04-28 16:36'],
              ['2026-04-29 08:52', '2026-04-29 11:45', '2026-04-29 15:45'],
              ['2026-04-30 07:58', '2026-04-30 12:47', '2026-04-30 16:47'],
              ['2026-05-01 08:40', '2026-05-01 10:38', '2026-05-01 14:38'],
              ['2026-05-04 08:16', '2026-05-04 10:10', '2026-05-04 14:11'],
              ['2026-05-05 07:40', '2026-05-05 12:55', '2026-05-05 16:59'],
              ['2026-05-06 07:12', '2026-05-06 11:16', '2026-05-06 15:23'],
              ['2026-05-07 11:06', '2026-05-07 13:41', '2026-05-07 17:42'],
              ['2026-05-08 07:59', '2026-05-08 11:20', '2026-05-08 15:20']]


# Read data
data = import_PTRMS([''], parent_path, True)

dict_keys = ['2026-04_VL+UV_dry_TS']*2 + ['2026-04_VL+UV_RH70_TS']*2 + ['2026-04_VL+UV_RH85_TS']*2 + ['2026-05_VL+UV_dry_TS']*2 + ['2026-05_VL+UV_RH85_seeds']*2 + ['2026-05_VL+UV_dry_seeds_TS']*2
Relative_humidity = ['Dry', 'Dry', '70% RH', '70% RH', '85% RH', '85% RH', 'Dry', 'Dry', '85% RH', '85% RH', 'Dry', 'Dry']

for ts, key, RH, folder in zip(timestamps[:3], dict_keys[:3], Relative_humidity[:3], Fig_folders[:3]):
    frag_keys = data[key].keys()[:3].tolist()
    parent_key = data[key].keys()[3]
    temp = time_filtered_conc(data[key], frag_keys+[parent_key], [ts[0], ts[2]])
    fit_ts = [pd.to_datetime(ts[1]) + pd.Timedelta(hours = 1), pd.to_datetime(ts[2]) - pd.Timedelta(minutes = 10)]

    labels = ['C$_{8}$H$_{8}$O$_{3}$H$^{+}$', 'C$_{5}$H$_{4}$H$^{+}$', 'C$_{6}$H$_{6}$O$_{2}$H$^{+}$', 
              'C$_{7}$H$_{8}$O$_{2}$H$^{+}$']

    fig, ax = plot_PTRMS_decay(temp, parent_key, frag_keys, labels, ts[1], fit_ts, None, RH)

    fig.tight_layout()
    fig.savefig(f'{save_path}{folder}/{ts[0].split(' ')[0]}_VL-decay_IDA.jpg', dpi = 600)

for ts, key, RH, folder in zip(timestamps[3:], dict_keys[3:], Relative_humidity[3:], Fig_folders[3:]):
    frag_keys = data[key].keys()[:3].tolist()
    parent_key = data[key].keys()[3]

    temp = time_filtered_conc(data[key], frag_keys+[parent_key], [ts[0], pd.to_datetime(ts[2]) + pd.Timedelta(hours = 1)])
    fit_ts = [pd.to_datetime(ts[1]) + pd.Timedelta(hours = 1), pd.to_datetime(ts[2]) - pd.Timedelta(minutes = 10)]
    exp_end_ts = [pd.to_datetime(ts[2]) + pd.Timedelta(minutes = 10), pd.to_datetime(ts[2]) + pd.Timedelta(minutes = 50)]

    labels = ['C$_{8}$H$_{8}$O$_{3}$H$^{+}$', 'C$_{5}$H$_{4}$H$^{+}$', 'C$_{6}$H$_{6}$O$_{2}$H$^{+}$', 
              'C$_{7}$H$_{8}$O$_{2}$H$^{+}$']

    fig, ax = plot_PTRMS_decay(temp, parent_key, frag_keys, labels, ts[1], fit_ts, exp_end_ts, RH)

    fig.tight_layout()
    fig.savefig(f'{save_path}{folder}/{ts[0].split(' ')[0]}_VL-decay_IDA.jpg', dpi = 600)
#%%
for ts, key in zip(timestamps, dict_keys):
    df_keys = data[key].keys()[:4].tolist()
    temp = time_filtered_conc(data[key], df_keys, [pd.to_datetime(ts[1]) - pd.Timedelta(minutes = 5), ts[1]])
    temp['Total conc'] = temp[df_keys].sum(axis = 1)
    start_conc, start_std = temp['Total conc'].mean(), temp['Total conc'].std()
    print(f'{ts[1].split(' ')[0]}: [VL]_0 = {start_conc:.1f} +- {start_std:.2f} ppb')
    