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
path = '../../../Data/2026/260422_Vanillin70ppb_UV_dry/'
SMPS = import_SMPS(f'{path}SMPS/', '', 0)
save_path = 'Figures/260422_vanillin+UV_dry/'
timestamps = [['2026-04-22 09:07', '2026-04-22 16:14']]

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)
#%%
SMPS_keys = [['260422_vanillin+UV_dry_number'], ['260422_vanillin+UV_dry_mass']]
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260422_vanillin+UV_dry_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, 'Total concentration', '2026-04-22 10:42', 1, 1, save_path)