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
# Path to data folder
parent_path = '../../../Data/2026/2026_TOTeM-campaign/'
# Path to figures
savepath = '../../../Figures/TOTeM/PTRMS/'

# Import PTR-MS data
PTRMS = import_PTRMS([''], parent_path)

# Import timestamps
timestamps = import_data(parent_path, '', None, None, None)
timestamps = timestamps['Timestamps'].replace(0, np.nan)

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

# Calculate start concentration of VOC
start_VOC = []
start_VOC_STD = []
for i, key in enumerate(exp_keys):
    time = [f'{timestamps['date'][i]} {timestamps['stabilization start'][i]}',
            f'{timestamps['date'][i]} {timestamps['stabilization end'][i]}']
    if '3MC' in key:
        temp = time_filtered_conc(PTRMS[key], ['3MC (ppb)'], time)
    if 'o-Cresol' in key:
            temp = time_filtered_conc(PTRMS[key], ['o-Cresol (ppb)'], time)
    if 'Toluene' in key:
            temp = time_filtered_conc(PTRMS[key], ['Toluene (ppb)'], time)
    start_VOC.append(temp[temp.keys()[1]].mean())
    start_VOC_STD.append(temp[temp.keys()[1]].std())
    print(f'{key.split('_')[0]}: {key.split('_')[1]} = ({temp[temp.keys()[1]].mean()} +- {temp[temp.keys()[1]].std()}) ppb')

# 1 min running mean of data
PTRMS_STD = {}
for dict_key in PTRMS.keys():
    df_keys = PTRMS[dict_key].keys()[:-2].tolist() + [PTRMS[dict_key].keys()[-1]]
    temp, temp_std = running_mean(PTRMS[dict_key], df_keys, 'Time', '1min', None)
    temp['Time'], temp_std['Time'] = temp.index, temp_std.index
    PTRMS[dict_key], PTRMS_STD[dict_key] = temp.reset_index(drop = True), temp_std.reset_index(drop = True)
#%%
# Max particle concentrations
max_particle_concentration = [17.79, 39.36, 0.00, 0.00, 0.01, 38.86, 25.66, 39.10, 24.67, 21.44, 68.23]
Mws = [124.14, 124.14, 92.14, 92.14, 92.14, 108.14, 108.14, 108.14, 108.14, 108.14, 124.14]
temperatures = [20, 20, 0, 20, 0, 0, 20, 0, 20, 20, 20]
#%%
for i, key in enumerate(exp_keys):
    t_zero = f'{timestamps['date'][i]} {timestamps['UV on'][i]}'
    fig, ax = plt.subplots(figsize = (6.3, 3))
    plot_total(ax, PTRMS[key], PTRMS[key].keys()[-2], None, t_zero)

    # UV on
    ax.axvspan(0, (pd.to_datetime(f'{timestamps['date'][i]} {timestamps['UV off'][i]}') -  pd.to_datetime(t_zero)) / pd.Timedelta(minutes = 1), color = 'yellow', alpha = 0.15, lw = 0, label = 'UV ov')

    # d9-BuOH inj
    ax.axvspan((pd.to_datetime(f'{timestamps['date'][i]} {timestamps['d9-BuOH inj start'][i]}') -  pd.to_datetime(t_zero)) / pd.Timedelta(minutes = 1), (pd.to_datetime(f'{timestamps['date'][i]} {timestamps['d9-BuOH inj end'][i]}') -  pd.to_datetime(t_zero)) / pd.Timedelta(minutes = 1), color = 'blue', alpha = 0.15, lw = 0, label = 'd9-BuOH inj')

    # H2O2 inj
    ax.axvspan((pd.to_datetime(f'{timestamps['date'][i]} {timestamps['H2O2 inj start'][i]}') -  pd.to_datetime(t_zero)) / pd.Timedelta(minutes = 1), (pd.to_datetime(f'{timestamps['date'][i]} {timestamps['H2O2 inj end'][i]}') -  pd.to_datetime(t_zero)) / pd.Timedelta(minutes = 1), color = 'orange', alpha = 0.15, label = 'H$_{2}$O$_{2}$ inj')

    # VOC inj
    ax.axvspan((pd.to_datetime(f'{timestamps['date'][i]} {timestamps['VOC inj start'][i]}') -  pd.to_datetime(t_zero)) / pd.Timedelta(minutes = 1), (pd.to_datetime(f'{timestamps['date'][i]} {timestamps['VOC inj end'][i]}') -  pd.to_datetime(t_zero)) / pd.Timedelta(minutes = 1), color = 'green', alpha = 0.15, lw = 0, label = 'VOC inj')

    ax.legend(frameon = True)

    ax.set(ylabel = 'VOC conc. (ppb)', title = f'{timestamps['date'][i]} {key.split('_')[1]}')

    fig.tight_layout()
    fig.savefig(f'{savepath}{timestamps['date'][i]}_{key.split('_')[1]}.jpg', dpi = 600)
#%%
def convert_ppb_to_ugm3(conc_ppb, Mw, T):
    conc_ppm = conc_ppb * 10**(-3)
    conc_ugm3 = ((1.013 * 10**(5) * Mw) / (8.314 * (T + 273))) * conc_ppm
    return conc_ugm3

def SOA_yield(VOC_start, VOC_max_mass, particle_max_mass):
     # VOC_start and VOC_max_mass has to be in um/m3
     return particle_max_mass / (VOC_start - VOC_max_mass)

for key, row, max_particle, VOC_start_conc, Mw, T in zip(exp_keys, timestamps.iterrows(), max_particle_concentration, start_VOC, Mws, temperatures):
    max_mass_ts = pd.to_datetime(f'{row[1]['date']} {row[1]['max particle conc']}')
    VOC_end = PTRMS[key][PTRMS[key].keys()[-2]][PTRMS[key]['Time'] == max_mass_ts]
    VOC_start_ugm3, VOC_end_ugm3 = convert_ppb_to_ugm3(VOC_start_conc, Mw, T), convert_ppb_to_ugm3(VOC_end, Mw, T)
    Y_SOA = SOA_yield(VOC_start_ugm3, VOC_end_ugm3, max_particle)
    print(f'{row[1]['date']} Y({key.split('_')[1]}) = {Y_SOA * 100}')