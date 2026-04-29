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
paths = ['260427_Vanillin70ppb_UV_RH70/', '260428_Vanillin70ppb_UV_RH70/'
]

SMPS = {}
for path in paths:
    temp = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp.keys():
        SMPS[key] = temp[key]

save_path = 'Figures/2604_vanillin+UV_RH70/'
timestamps = [['2026-04-27 10:36', '2026-04-27 17:28'],
              ['2026-04-28 11:07', '2026-04-28 17:38']
              ]
t_zero = ['2026-04-27 12:16', '2026-04-28 12:35'
]

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)
#%%
SMPS_keys = [['260427_vanillin+UV_RH70_number', '260428_vanillin+UV_RH70_number'],
             ['260427_vanillin+UV_RH70_mass', '260428_vanillin+UV_RH70_mass']]
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260428_vanillin+UV_RH70_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, 'Total concentration', t_zero, 1, 2, save_path)