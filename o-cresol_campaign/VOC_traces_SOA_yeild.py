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
parent_path = '../../../Data/2026/2026_TOTeM-campaign/'

PTRMS = import_PTRMS([''], parent_path)

print(PTRMS.keys())

PTRMS['2026-06-30_3MC_traces']['3MC (ppb)'] = PTRMS['2026-06-30_3MC_traces'].iloc[:, :-1].sum(axis = 1)
PTRMS['2026-07-14_o-Cresol_traces']['o-Cresol 110 (ppb)'] = PTRMS['2026-07-14_o-Cresol_traces'].iloc[:, [0, 2, 3]].sum(axis = 1)
PTRMS['2026-07-14_o-Cresol_traces']['o-Cresol 111 (ppb)'] = PTRMS['2026-07-14_o-Cresol_traces'].iloc[:, 1:-2].sum(axis = 1)

df_keys = [['3MC (ppb)'], ['o-Cresol 110 (ppb)', 'o-Cresol 111 (ppb)']]
for dict_key, df_key in zip(PTRMS.keys(), df_keys):
    temp, temp_std = running_mean(PTRMS[dict_key], df_key, 'Time', '1min', None)
    for key in temp_std.keys():
        temp[f'STD {key}'] = temp_std[key]
    temp['Time'] = temp.index
    PTRMS[dict_key] = temp.reset_index(drop = True)
    print(PTRMS[dict_key])

#%%
fig_3MC, ax_3MC = plt.subplots(figsize = (6.3, 3))
plot_total(ax_3MC, PTRMS['2026-06-30_3MC_traces'], '3MC (ppb)', 'b', None)
ax_3MC.set(ylabel = '3MC conc (ppb)')
fig_3MC.tight_layout()

fig_Cresol, ax_Cresol = plt.subplots(figsize = (6.3, 3))
plot_total(ax_Cresol, PTRMS['2026-07-14_o-Cresol_traces'], 'o-Cresol 110 (ppb)', 'b', None)
plot_total(ax_Cresol, PTRMS['2026-07-14_o-Cresol_traces'], 'o-Cresol 111 (ppb)', 'r', None)
ax_Cresol.legend(['o-Cresol w. m/z 110', 'o-Cresol w. m/z 111'])
ax_Cresol.set(ylabel = 'o-Cresol conc (ppb)')
fig_Cresol.tight_layout()
# %%
stabilization_time = [['2026-06-30 14:00', '2026-06-30 14:10'],
                      ['2026-07-14 11:44', '2026-07-14 11:54']]

for dict_key, df_key, time in zip(PTRMS.keys(), df_keys, stabilization_time):
    temp = time_filtered_conc(PTRMS[dict_key], df_key, time)
    for key in df_key:
        conc = temp[key].mean()
        print(f'{key}: {conc} ppb')