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
paths = ['20260623_cresol_test_H2O2_UV/']

timestamps = [['2026-06-23 11:15', '2026-06-23 15:22']]
t_zero = ['2026-06-23 13:20']

SMPS = {}
for path in paths:
    temp_SMPS = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp_SMPS.keys():
        SMPS[key] = temp_SMPS[key]

save_path = '../../../Figures/o-cresol_campaign/'

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)

SMPS_keys = [['260623_cresol_test_H2O2_UV_number'], 
             ['260623_cresol_test_H2O2_UV_mass']]
#%%
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260623_cresol_test_H2O2_UV_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, 'Dry', 'Total concentration', t_zero, 1, 1, save_path)