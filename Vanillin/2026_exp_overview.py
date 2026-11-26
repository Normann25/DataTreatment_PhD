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
SMPS = import_SMPS([''], path, 0, [52]*13 + [0]*20 + [52]*16)
# PTRMS = import_PTRMS([''], path)
AMS = import_AMS([''], path, 0)
DAQ = import_DAQ([''], path, 0)

# Remove spikes from data
for i, key in enumerate(DAQ_keys):
    # SMPS
    for row, min_conc in zip(SMPS_keys, [10, 0.01]):
        temp = SMPS[row[i]]
        temp.loc[temp['Total concentration'] <= min_conc, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        SMPS[row[i]] = remove_spikes_up(temp, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 20)
    # DAQ
    DAQ[key] = remove_spikes(DAQ[key], ['Temp_C'], 5)
    if 'dry' in SMPS_keys[0][i]:
        RH_mask = DAQ[key]['RH_Percent'] < 0.005
        DAQ[key] = DAQ[key][RH_mask]

RH = ['dry', '85% RH', 'seeds, dry', 'seeds, 85% RH']

fig, ax = plot_AURA_overview(DAQ, DAQ_keys, SMPS, SMPS_keys[0], AMS, AMS_keys, 
                             t_exp, t_HEPA, t_UV_on, 4, 4, RH, [(5, 100), (-1, 2)], save_path)
#%%

timestamps = [['2026-01-13 14:36', '2026-01-13 18:25'],
              ['2026-01-15 11:10', '2026-01-15 17:10'],
              ['2026-01-16 12:44', '2026-01-16 17:56'],
              ['2026-04-21 11:28', '2026-04-21 17:01'],
              ['2026-04-22 10:42', '2026-04-22 16:14'],
              ['2026-04-23 11:58', '2026-04-23 15:59'],
              ['2026-04-27 12:16', '2026-04-27 16:21'],
              ['2026-04-28 12:35', '2026-04-28 16:36'],
              ['2026-04-29 11:45', '2026-04-29 15:45'],
              ['2026-04-30 12:47', '2026-04-30 16:47'],
              ['2026-05-01 10:38', '2026-05-01 14:38'],
              ['2026-05-04 10:10', '2026-05-04 14:11'],
              ['2026-05-05 12:55', '2026-05-05 16:59'],
              ['2026-05-06 11:16', '2026-05-06 15:23'],
              ['2026-05-07 13:41', '2026-05-07 17:42'],
              ['2026-05-08 11:20', '2026-05-08 15:20']]

SMPS_number = ['260113-vanillin_number', '260115-vanillin+UV_number', '260116_vanillin+apinene+UV_number',
               '260421_vanillin+UV_dry_number', '260422_vanillin+UV_dry_number', '260423_Vanillin70ppb_UV_RH70_number',
               '260427_vanillin+UV_RH70_number', '260428_vanillin+UV_RH70_number', '260429_vanillin+UV_RH85_number', '260430_vanillin+UV_RH85_number', '260501_vanillin+UV_dry_number', '260504_vanillin+UV_dry_number',
               '260505_vanillin+UV+seeds_RH85_number', '260506_vanillin+UV+seeds_RH85_number', 
               '260507_vanillin+UV+seeds_dry_number', '260508_vanillin+UV+seeds_dry_number']

SMPS_mass = ['260113-vanillin_mass', '260115-vanillin+UV_mass', '260116_vanillin+apinene+UV_mass',
             '260421_vanillin+UV_dry_mass', '260422_vanillin+UV_dry_mass', '260423_Vanillin70ppb_UV_RH70_mass',
             '260427_vanillin+UV_RH70_mass', '260428_vanillin+UV_RH70_mass', '260429_vanillin+UV_RH85_mass', '260430_vanillin+UV_RH85_mass', '260501_vanillin+UV_dry_mass', '260504_vanillin+UV_dry_mass', 
             '260505_vanillin+UV+seeds_RH85_mass', '260506_vanillin+UV+seeds_RH85_mass', 
             '260507_vanillin+UV+seeds_dry_mass', '260508_vanillin+UV+seeds_dry_mass']

for number_key, mass_key, t in zip(SMPS_number, SMPS_mass, timestamps):
    number = time_filtered_conc(SMPS[number_key], ['Total concentration'], t)
    mass = time_filtered_conc(SMPS[mass_key], ['Total concentration'], t)

    t_max_number = number.loc[number['Total concentration'] == max(number['Total concentration']), 'Time']
    t_max_number = pd.to_datetime(t_max_number.iloc[0])
    t_max_number = [t_max_number - pd.Timedelta(minutes = 5), t_max_number + pd.Timedelta(minutes = 5)]
    number = time_filtered_conc(number, ['Total concentration'], t_max_number)
    max_number, number_std = number['Total concentration'].mean(), number['Total concentration'].std()

    t_max_mass = mass.loc[mass['Total concentration'] == max(mass['Total concentration']), 'Time']
    t_max_mass = pd.to_datetime(t_max_mass.iloc[0])
    t_max_mass = [t_max_mass - pd.Timedelta(minutes = 5), t_max_mass + pd.Timedelta(minutes = 5)]
    mass = time_filtered_conc(mass, ['Total concentration'], t_max_mass)
    max_mass, mass_std = mass['Total concentration'].mean(), mass['Total concentration'].std()

    print(f'{t[0].split(' ')[0]}')
    print(f'Max number concentration: N_max = {max_number:.1f} +- {number_std:.2f} cm^-3')
    print(f'Max mass concentration: M_max = {max_mass:.2f} +- {mass_std:.4f} ug m^-3')

    daq_key = f'DataDAQ_26{t[0].split('-')[1]}{t[0].split('-')[2].split(' ')[0]}'
    daq = time_filtered_conc(DAQ[daq_key], ['RH_Percent'], t)
    RH_mean, RH_std = daq['RH_Percent'].mean(), daq['RH_Percent'].std()
    print(f'RH = {RH_mean:.1f} +- {RH_std:.3f}')
