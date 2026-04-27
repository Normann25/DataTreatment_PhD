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
path = '../../../Data/2026/260427_Vanillin70ppb_UV_RH70/'
SMPS = import_SMPS(f'{path}SMPS/', '', 0)
save_path = 'Figures/260427_vanillin+UV_RH70/'
timestamps = [['2026-04-27 10:36', '2026-04-27 17:27']]

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)
#%%
SMPS_keys = [['260427_vanillin+UV_RH70_number'], ['260427_vanillin+UV_RH70_mass']]
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260427_vanillin+UV_RH70_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, 'Total concentration', '2026-04-27 12:16', 1, 1, save_path)