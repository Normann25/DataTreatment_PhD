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
paths = ['260421_Vanillin70ppb_UV_dry_v2/', '260422_Vanillin70ppb_UV_dry/', '260501_Vanillin70ppb_UV_dry/', '260504_Vanillin70ppb_UV_dry/']

timestamps = [['2026-04-21 09:11', '2026-04-21 17:01'],
              ['2026-04-22 08:50', '2026-04-22 16:14'],
              ['2026-05-01 10:17', '2026-05-01 15:38'],
              ['2026-05-04 08:45', '2026-05-04 15:11']]
t_zero = ['2026-04-21 11:28', '2026-04-22 10:42', '2026-05-01 10:38', '2026-05-04 10:10']
t_UV_off = ['2026-05-01 14:38', '2026-05-04 14:11']

# bg_timestamps = [['2026-04-21 10:35', '2026-04-21 10:58'],
#                  ['2026-04-22 08:45', '2026-04-22 09:06']]
# SMPS = {}
# for path in paths:
#     temp = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
#     for bg_time, key in zip(bg_timestamps, temp.keys()):
#         background = time_filtered_conc(temp[key], [temp[key].keys()[38]]+temp[key].keys()[42:-1].to_list(), bg_time)
#         for df_key in [temp[key].keys()[38]]+temp[key].keys()[42:-1].to_list():
#             temp[key][df_key] = temp[key][df_key] - background[df_key].mean()
#         SMPS[key] = temp[key]

SMPS = {}
PTRMS = {}
for t, path in zip(t_zero, paths):
    temp = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp.keys():
        temp[key].loc[temp[key]['Time'] < pd.to_datetime(t), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        temp[key].loc[temp[key]['Geo. Mean (nm)'] > 80, ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        SMPS[key] = temp[key]
    temp_PTR = import_PTRMS(f'{parent_path}{path}PTRMS/', '')
    for key in temp_PTR.keys():
        mask = (0 < temp_PTR[key][temp_PTR[key].keys()[5]]) & (temp_PTR[key][temp_PTR[key].keys()[5]] < 90)
        temp = temp_PTR[key][mask]
        PTRMS[key] = temp

AMS = import_data(f'{parent_path}{paths[1]}AMS/', '', 't_series', '%d-%m-%Y %H:%M:%S', 0)
for key in AMS.keys():            
    AMS[key].columns = ['t_series', 'HROrg', 'HRNO3', 'HRSO4', 'HRNH4', 'HRChl', 'Ratio_H_C', 'Ratio_O_C', 
                        'familyCHN', 'familyCHO1', 'familyCHO1N', 'familyCH', 'f43', 'f44', 'Time']

save_path = 'Figures/2604_vanillin+UV_dry/'

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)
#%%
SMPS_keys = [['260421_vanillin+UV_dry_number', '260422_vanillin+UV_dry_number', '260501_vanillin+UV_dry_number', '260504_vanillin+UV_dry_number'], 
             ['260421_vanillin+UV_dry_mass', '260422_vanillin+UV_dry_mass', '260501_vanillin+UV_dry_mass', '260504_vanillin+UV_dry_mass']]
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260422_vanillin+UV_dry_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, ['Dry']*len(t_zero), 'Total concentration', t_zero, 2, 2, save_path)
#%%
bg_timestamps = ['2026-04-22 08:25', '2026-04-22 08:40']
plot_AMS(AMS['260422_AMS_vanillin+UV_dry_TS'], None, t_zero[1], timestamps[1], bg_timestamps, 1, save_path)
#%%
PTRMS_keys = ['260501_VL+UV_dry_initial', '260504_VL+UV_dry_initial']
for i, key in enumerate(PTRMS_keys):
    fig, ax = plot_PTRMS_decay(PTRMS[key], PTRMS[key].keys()[5], [PTRMS[key].keys()[4]], 
                               ['C$_{8}$H$_{9}$O$_{3}^{+}$', 'C$_{7}$H$_{9}$O$_{2}^{+}$'], 
                               t_zero[i+2], t_UV_off[i], timestamps[i+2][1], 'Dry')
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+1].split(' ')[0]}_PTRMS_initial.jpg', dpi = 600)