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
# Import PTR-MS data
parent_path = '../../../Data/2026/2026_TOTeM-campaign/'
PTRMS = import_PTRMS([''], parent_path)

# Calculate total concentration of VOC from parent compound and fragments
for key in PTRMS.keys():
    if '3MC' in key:
        PTRMS[key]['3MC (ppb)'] = PTRMS[key].loc[:, ['m107.050 (C[12]7H[1]7O[16]1) (Conc)', 'm124.049 (C[12]6C[13]1H[1]7O[16]2) (Conc)', 'm126.063 (C[12]6C[13]1H[1]9O[16]2) (Conc)', 'm139.040 (C[12]7H[1]7O[16]3) (Conc)', 'm151.076 (C[12]9H[1]11O[16]2) (Conc)']].sum(axis = 1)
    if 'o-Cresol' in key:
        PTRMS[key]['o-Cresol (ppb)'] = PTRMS[key].loc[:, ['m110.069 (C[12]6C[13]1H[1]9O[16]1) (Conc)', 'm123.045 (C[12]7H[1]7O[16]2) (Conc)', 'm125.055 (C[12]7H[1]9O[16]2) (Conc)']].sum(axis = 1)
    if 'Toluene' in key:
        PTRMS[key]['Toluene (ppb)'] = PTRMS[key].iloc[:, :-2].sum(axis = 1)

# Names of experiments
exp_keys = ['20260629_3MC_traces', '20260630_3MC_traces', '20260707_Toluene_traces', '20260708_Toluene_traces',
            '20260709_Toluene_traces', '20260710_o-Cresol_traces', '20260711_o-Cresol_traces', '20260713_o-Cresol_traces',
            '20260714_o-Cresol_traces', '20260715_o-Cresol_traces', '20260716_3MC_traces']
blank_keys = ['20260730_o-Cresol_traces', '20260731_3MC_traces']

# Define experiment timestamps
stabilization_time = [['2026-06-29 13:22', '2026-06-29 13:27'],
                      ['2026-06-30 14:00', '2026-06-30 14:10'],
                      ['2026-07-07 11:40', '2026-07-07 11:45'],
                      ['2026-07-08 11:40', '2026-07-08 11:45'],
                      ['2026-07-09 10:43', '2026-07-09 10:48'],
                      ['2026-07-10 11:16', '2026-07-10 11:21'],
                      ['2026-07-11 12:13', '2026-07-11 12:18'],
                      ['2026-07-13 12:21', '2026-07-13 12:26'],
                      ['2026-07-14 11:44', '2026-07-14 11:54'],
                      ['2026-07-15 11:56', '2026-07-15 12:01'],
                      ['2026-07-16 11:30', '2026-07-16 11:35']]
UV_on = ['2026-06-29 13:28', '2026-06-30 14:12', ' 2026-07-07 11:36', '2026-07-08 11:50', '2026-07-09 10:49', 
         '2026-07-10 11:22', ' 2026-07-11 12:19', '2026-07-13 12:27', '2026-07-14 11:55', '2026-07-15 12:02',
         '2026-07-16 11:36']
max_particle_time = ['2026-06-29 15:22', '2026-06-30 16:18', ' 2026-07-07 13:45', '2026-07-08 13:02', '2026-07-09 11:55', 
                     '2026-07-10 14:55', ' 2026-07-11 15:06', '2026-07-13 17:40', '2026-07-14 16:12', '2026-07-15 15:45',
                     '2026-07-16 13:08']

start_VOC = []
start_VOC_STD = []
for key, time in zip(exp_keys, stabilization_time):
    if '3MC' in key:
        temp = time_filtered_conc(PTRMS[key], ['3MC (ppb)'], time)
    if 'o-Cresol' in key:
            temp = time_filtered_conc(PTRMS[key], ['o-Cresol (ppb)'], time)
    if 'Toluene' in key:
            temp = time_filtered_conc(PTRMS[key], ['Toluene (ppb)'], time)
    start_VOC.append(temp[temp.keys()[1]].mean())
    start_VOC_STD.append(temp[temp.keys()[1]].std())
    print(f'{key.split('_')[0]}: {key.split('_')[1]} = ({temp[temp.keys()[1]].mean()} +- {temp[temp.keys()[1]].std()}) ppb')

# Max particle concentrations
max_particle_concentration = [17.79, 39.36, 0.00, 0.00, 0.01, 38.86, 25.66, 39.10, 24.67, 21.44, 68.23]

# 1 min running mean of data
PTRMS_STD = {}
for dict_key in PTRMS.keys():
    df_keys = PTRMS[dict_key].keys()[:-2].tolist() + [PTRMS[dict_key].keys()[-1]]
    temp, temp_std = running_mean(PTRMS[dict_key], df_keys, 'Time', '1min', None)
    temp['Time'], temp_std['Time'] = temp.index, temp_std.index
    PTRMS[dict_key], PTRMS_STD[dict_key] = temp.reset_index(drop = True), temp_std.reset_index(drop = True)
#%%
for key in exp_keys:
    fig, ax = plt.subplots()
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

fig_Toluene, ax_Toluene = plt.subplots(figsize = (6.3, 3))
plot_total(ax_Toluene, PTRMS['2026-07-08_Toluene_traces'], 'Toluene (ppb)', 'b', None)
ax_Toluene.set(ylabel = 'Toluene conc (ppb)')
fig_Toluene.tight_layout()
# %%
stabilization_time = [['2026-06-30 14:00', '2026-06-30 14:10'],
                      ['2026-07-08 11:40', '2026-07-08 11:45'],
                      ['2026-07-14 11:44', '2026-07-14 11:54']]

for dict_key, df_key, time in zip(PTRMS.keys(), df_keys, stabilization_time):
    PTRMS[dict_key].to_csv(f'{parent_path}{dict_key.split('_') [1]}.csv', index = False)
    temp = time_filtered_conc(PTRMS[dict_key], df_key, time)
    for key in df_key:
        conc = temp[key].mean()
        print(f'{key}: {conc} +- {temp[key].std()}')
        