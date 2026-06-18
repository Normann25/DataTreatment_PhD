#%%
import sys
sys.path.append('../')
from read_data_functions import *
from plot_functions import *
from calculations import *
from grouping import *
from scipy.signal import find_peaks
plt.style.use('../Style.mplstyle')
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # suppress warnings
#%%
parent_path = '../../../Data/2026/'
paths = ['260421_Vanillin70ppb_UV_dry_v2/', '260422_Vanillin70ppb_UV_dry/', '260501_Vanillin70ppb_UV_dry/', '260504_Vanillin70ppb_UV_dry/']

timestamps = [['2026-04-21 09:11', '2026-04-21 17:01'],
              ['2026-04-22 08:50', '2026-04-22 16:14'],
              ['2026-05-01 10:17', '2026-05-01 15:38'],
              ['2026-05-04 08:45', '2026-05-04 15:11']]
t_zero = ['2026-04-21 11:28', '2026-04-22 10:42', '2026-05-01 10:38', '2026-05-04 10:10']
t_UV_off = ['2026-05-01 14:38', '2026-05-04 14:11']
HEPA_timestamps = [['2026-04-21 08:40', '2026-04-21 09:00'],
                   ['2026-04-22 08:25', '2026-04-22 08:40'],
                   ['2026-05-01 08:50', '2026-05-01 09:10'],
                   ['2026-05-04 08:10', '2026-05-04 08:30']]

def remove_spikes(df, df_keys, value):
    """
    Parameters
    ----------
    df: pandas dataframe

    df_keys: dataframe keys

    value: float or int
        Peak detection threshold
    """

    for key in df_keys:
        # `prominence` sets minimum height above surrounding 
        # signal at which a given value is considered a peak
        peak_idx = find_peaks(df[key], prominence=value)[0]

        # To detect valleys deeper than `value`, 
        # run find_peaks on negative of data
        valley_idx = find_peaks(-df['Admissions to Date'], prominence=value)[0]

        # Combine indexes of peaks and valleys into a single array
        idx = np.concatenate((peak_idx, valley_idx))

        # Build an indicator column of peaks and valleys, or outliers
        df['outlier'] = False
        df.loc[idx, 'outlier'] = True

        # Replace each outlier value with NaN
        df.loc[df['outlier'], key] = np.nan

        # Interpolate over NaNs just created with default linear method
        df[key] = (df[key].interpolate().astype(float))

        df = df.drop(['outlier'], axis = 1)

    return df

SMPS = {}
PTRMS = {}
AMS = {}
DAQ = {}
for t, path in zip(t_zero, paths):
    temp_smps = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp_smps.keys():
        temp = remove_spikes(temp_smps[key], [temp_smps[key].keys()[38]], max(temp_smps[key][temp_smps[key].keys()[38]])/10)
        temp.loc[temp['Time'] < pd.to_datetime(t), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = 0
        temp.loc[temp[temp_smps[key].keys()[38]] == 0, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = 0
        temp.loc[temp['Geo. Mean (nm)'] > 100, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = 0
        temp = remove_spikes(temp, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)'], 10)
        SMPS[key] = temp
    temp_PTR = import_PTRMS(f'{parent_path}{path}PTRMS/', '')
    for key in temp_PTR.keys():
        mask = (0 < temp_PTR[key][temp_PTR[key].keys()[5]]) & (temp_PTR[key][temp_PTR[key].keys()[5]] < 90)
        temp = temp_PTR[key][mask]
        PTRMS[key] = temp
    temp_AMS = import_data(f'{parent_path}{path}AMS/', '', 't_series', '%d-%m-%Y %H:%M:%S', 0)
    for key in temp_AMS.keys():
        if 'PToF' not in key:
            temp_AMS[key].columns = ['t_series', 'HROrg', 'HRNO3', 'HRSO4', 'HRNH4', 'HRChl', 'Ratio_H_C', 'Ratio_O_C', 
                            'familyCHN', 'familyCHO1', 'familyCHO1N', 'familyCH', 'f43', 'f44', 'Time']
            mask = temp_AMS[key]['HROrg'] != 0
            temp_AMS[key] = temp_AMS[key][mask]
        AMS[key] = temp_AMS[key]
    temp_daq = import_data(f'{parent_path}{path}DAQ/', '', 'DAQ_Timestamp_UTC', '%d-%m-%Y %H:%M:%S', 0)
    for key in temp_daq.keys():
        DAQ[key] = temp_daq[key]

save_path = 'Figures/2604_vanillin+UV_dry/'

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)

SMPS_keys = [['260421_vanillin+UV_dry_number', '260422_vanillin+UV_dry_number', '260501_vanillin+UV_dry_number', '260504_vanillin+UV_dry_number'], 
             ['260421_vanillin+UV_dry_mass', '260422_vanillin+UV_dry_mass', '260501_vanillin+UV_dry_mass', '260504_vanillin+UV_dry_mass']]
AMS_keys = ['260421_AMS_vanillin+UV_dry_TS', '260422_AMS_vanillin+UV_dry_TS', '260501_AMS_vanillin+UV_dry_TS', '260504_AMS_vanillin+UV_dry_TS']
PTRMS_keys = ['260501_VL+UV_dry_initial', '260504_VL+UV_dry_initial']
DAQ_keys = ['DataDAQ_260421', 'DataDAQ_260422', 'DataDAQ_260501', 'DataDAQ_260504']
#%%
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260422_vanillin+UV_dry_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, ['Dry']*len(t_zero), 'Total concentration', t_zero, 2, 2, save_path)
#%%
for i, key in enumerate(AMS_keys):
    plot_AMS(AMS[key], None, t_zero[i], timestamps[i], HEPA_timestamps[i], 1, 'Dry', save_path)
#%%
for i, key in enumerate(PTRMS_keys):
    fig, ax = plot_PTRMS_decay(PTRMS[key], PTRMS[key].keys()[5], [PTRMS[key].keys()[4]], 
                               ['C$_{8}$H$_{9}$O$_{3}^{+}$', 'C$_{7}$H$_{9}$O$_{2}^{+}$'], 
                               t_zero[i+2], t_UV_off[i], timestamps[i+2][1], 'Dry')
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+1].split(' ')[0]}_PTRMS_initial.jpg', dpi = 600)
#%%
for i, time in enumerate(timestamps):
    plot_AURA_overview(DAQ[DAQ_keys[i]], SMPS[SMPS_keys[0][i]], AMS[AMS_keys[i]], time, t_zero[i], save_path)