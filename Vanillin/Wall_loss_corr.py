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
parent_path_SMPS = '../../../Data/2026/'
parent_path_AMS = '../../../Data/Processed data/2026_vanillin-campaign/'
paths = ['20260427_Vanillin70ppb_UV_RH70/', '20260428_Vanillin70ppb_UV_RH70/', 
         '20260429_Vanillin70ppb_UV_RH85/', '20260430_Vanillin70ppb_UV_RH85/',
         '20260501_Vanillin70ppb_UV_dry/', '20260504_Vanillin70ppb_UV_dry/',
         '20260505_Vanillin70ppb_UV_85RH_(NH4)2SO4-seeds/', '20260506_Vanillin70ppb_UV_85RH_(NH4)2SO4-seeds/',
         '20260507_Vanillin70ppb_UV_dry_(NH4)2SO4-seeds/', '20260508_Vanillin70ppb_UV_dry_(NH4)2SO4-seeds/']
save_path = '../../../Data/Processed data/2026_vanillin-campaign/'

# HEPA timestamps, UV on and off (for non-seeded), seed injection and UV off (seeded) timestamps 
timestamps = [['2026-04-27 08:40', '2026-04-27 09:00', '2026-04-27 12:16', '2026-04-27 16:21'],
              ['2026-04-28 07:55', '2026-04-28 08:15', '2026-04-28 12:35', '2026-04-28 16:36'],
              ['2026-04-29 08:40', '2026-04-29 09:00', '2026-04-29 11:45', '2026-04-29 15:45'],
              ['2026-04-30 08:15', '2026-04-30 08:35', '2026-04-30 12:47', '2026-04-30 16:47'],
              ['2026-05-01 08:50', '2026-05-01 09:10', '2026-05-01 10:38', '2026-05-01 14:38'],
              ['2026-05-04 08:10', '2026-05-04 08:30', '2026-05-04 10:10', '2026-05-04 14:11'],
              ['2026-05-05 07:47', '2026-05-05 08:07', '2026-05-05 12:27', '2026-05-05 16:59'],
              ['2026-05-06 07:26', '2026-05-06 07:46', '2026-05-06 10:45', '2026-05-06 15:23'],
              ['2026-05-07 08:10', '2026-05-07 08:30', '2026-05-07 13:06', '2026-05-07 17:42'],
              ['2026-05-08 08:13', '2026-05-08 08:33', '2026-05-08 10:49', '2026-05-08 15:20']]

# Dictionary keys
SMPS_keys = [['260427_vanillin+UV_RH70_number', '260428_vanillin+UV_RH70_number', 
              '260429_vanillin+UV_RH85_number', '260430_vanillin+UV_RH85_number',
              '260501_vanillin+UV_dry_number', '260504_vanillin+UV_dry_number',
              '260505_vanillin+UV+seeds_RH85_number', '260506_vanillin+UV+seeds_RH85_number', 
              '260507_vanillin+UV+seeds_dry_number', '260508_vanillin+UV+seeds_dry_number'],
             ['260427_vanillin+UV_RH70_mass', '260428_vanillin+UV_RH70_mass', 
              '260429_vanillin+UV_RH85_mass', '260430_vanillin+UV_RH85_mass',
              '260501_vanillin+UV_dry_mass', '260504_vanillin+UV_dry_mass',
              '260505_vanillin+UV+seeds_RH85_mass', '260506_vanillin+UV+seeds_RH85_mass', 
              '260507_vanillin+UV+seeds_dry_mass', '260508_vanillin+UV+seeds_dry_mass']]

AMS_keys = ['260427_AMS_vanillin+UV_70RH_TS', '260428_AMS_vanillin+UV_70RH_TS', 
            '260429_AMS_vanillin+UV_85RH_TS', '260430_AMS_vanillin+UV_85RH_TS',
            '260501_AMS_vanillin+UV_dry_TS', '260504_AMS_vanillin+UV_dry_TS',
            '260505_AMS_vanillin+UV+seeds_85RH_TS', '260506_AMS_vanillin+UV+seeds_85RH_TS', 
            '260507_AMS_vanillin+UV+seeds_dry_TS', '260508_AMS_vanillin+UV+seeds_dry_TS']
AMS_conc_keys = ['HROrg', 'HRNO3', 'HRSO4', 'HRNH4', 'HRChl',
                 'familyCHO1', 'familyCHOgt1', 'familyCH', 'familyCHN', 'familyCHO1N']

AMS_data = import_AMS(paths, parent_path_AMS, 0)
SMPS_data = import_SMPS(paths, parent_path_SMPS, 0)

for i, t in enumerate(timestamps):
    # SMPS
    temp_number, temp_mass = SMPS_data[SMPS_keys[0][i]], SMPS_data[SMPS_keys[1][i]]
    t_zero = pd.to_datetime(t[2])
    temp_number.loc[temp_number['Time'] < t_zero, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    temp_mass.loc[temp_mass['Time'] < t_zero, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    if i < 7:
        temp_number.loc[temp_number['Total concentration'] <= 10, 
                        ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        temp_mass.loc[temp_mass['Total concentration'] <= 0.05,
                      ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
    smps_number = wall_loss_corr(temp_number, ['Total concentration'], 
                                 t[2], [t[3], pd.to_datetime(t[3]) + pd.Timedelta( hours = 1)])
    smps_mass = wall_loss_corr(temp_mass, ['Total concentration'], 
                               t[2], [t[3], pd.to_datetime(t[3]) + pd.Timedelta( hours = 1)])
    smps_number.to_csv(f'{save_path}SMPS/{SMPS_keys[0][i]}.csv', index = False)
    smps_mass.to_csv(f'{save_path}SMPS/{SMPS_keys[1][i]}.csv', index = False)

    # AMS
    temp = AMS_data[AMS_keys[i]]
    bg_df = time_filtered_conc(temp, AMS_conc_keys, t[:2])
    UV_off_df = time_filtered_conc(temp, AMS_conc_keys, [t[3], pd.to_datetime(t[3]) + pd.Timedelta( hours = 1)])
    temp_keys = []
    for key in AMS_conc_keys:
        temp[key] = temp[key] - bg_df[key].mean()
        if UV_off_df[key].mean() > 0:
            temp_keys.append(key)

    ams = wall_loss_corr(temp, temp_keys, t[2], [t[3], pd.to_datetime(t[3]) + pd.Timedelta(hours = 1)])
    ams.rename(columns = {'Time': 't_series'}, inplace = True)
    ams.to_csv(f'{save_path}AMS/{AMS_keys[i]}.txt', sep = '\t', index = False)