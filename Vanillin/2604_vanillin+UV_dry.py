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
paths = ['260421_Vanillin70ppb_UV_dry_v2/', '260422_Vanillin70ppb_UV_dry/']

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
for path in paths:
    temp = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp.keys():
        SMPS[key] = temp[key]

save_path = 'Figures/2604_vanillin+UV_dry/'
timestamps = [['2026-04-21 10:59', '2026-04-21 17:01'],
              ['2026-04-22 09:07', '2026-04-22 16:14']
              ]
t_zero = ['2026-04-21 11:28', '2026-04-22 10:42'
]

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)
#%%
SMPS_keys = [['260421_vanillin+UV_dry_number', '260422_vanillin+UV_dry_number'], 
             ['260421_vanillin+UV_dry_mass', '260422_vanillin+UV_dry_mass']]
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260422_vanillin+UV_dry_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, 'Total concentration', t_zero, 1, 2, save_path)