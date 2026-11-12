#%%
import sys
sys.path.append('../')
from read_data_functions import *
from plot_functions import *
from calculations import *
from grouping import *

from sympy import*
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

# Import timestamps and density
data = import_data(parent_path, '', None, None, None)
data['Timestamps'] = data['Timestamps'].replace(0, np.nan)

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
    time = [f'{data['Timestamps']['date'][i]} {data['Timestamps']['stabilization start'][i]}',
            f'{data['Timestamps']['date'][i]} {data['Timestamps']['stabilization end'][i]}']
    if '3MC' in key:
        temp = time_filtered_conc(PTRMS[key], ['3MC (ppb)'], time)
    if 'o-Cresol' in key:
            temp = time_filtered_conc(PTRMS[key], ['o-Cresol (ppb)'], time)
    if 'Toluene' in key:
            temp = time_filtered_conc(PTRMS[key], ['Toluene (ppb)'], time)
    start_VOC.append(temp[temp.keys()[1]].mean())
    start_VOC_STD.append(temp[temp.keys()[1]].std())
    print(f'{key.split('_')[0]}: {key.split('_')[1]} = ({temp[temp.keys()[1]].mean():.2f} +- {temp[temp.keys()[1]].std():.2f}) ppb')

# 1 min running mean of data
PTRMS_STD = {}
for dict_key in PTRMS.keys():
    df_keys = PTRMS[dict_key].keys()[:-2].tolist() + [PTRMS[dict_key].keys()[-1]]
    temp, temp_std = running_mean(PTRMS[dict_key], df_keys, 'Time', '1min', None)
    temp['Time'], temp_std['Time'] = temp.index, temp_std.index
    PTRMS[dict_key], PTRMS_STD[dict_key] = temp.reset_index(drop = True), temp_std.reset_index(drop = True)
#%%
# Max particle concentrations
max_particle_concentration = np.array([17.79, 39.36, 0.00, 0.00, 0.01, 38.86, 25.66, 39.10, 24.67, 21.44, 68.23])
max_particle_concentration = max_particle_concentration * data['density_TOTEM']['Denisty'][::-1]
Mws = [124.14, 124.14, 92.14, 92.14, 92.14, 108.14, 108.14, 108.14, 108.14, 108.14, 124.14]
temperatures = [20, 20, 0, 20, 0, 0, 20, 0, 20, 20, 20]
# #%%
# for key, row, max_particle, VOC_start_conc, Mw, T in zip(exp_keys, data['Timestamps'].iterrows(), max_particle_concentration, start_VOC, Mws, temperatures):
#     # Determine time period where maximum particle mass is reached
#     max_mass_ts1 = pd.to_datetime(f'{row[1]['date']} {row[1]['max particle conc']}') - pd.Timedelta(minutes = 2) 
#     max_mass_ts2 = pd.to_datetime(f'{row[1]['date']} {row[1]['max particle conc']}') + pd.Timedelta(minutes = 2)

#     # Calculate VOC concentration when maximum particle mass is reached (from time-2min to time+2min)
#     VOC_end = time_filtered_conc(PTRMS[key], [PTRMS[key].keys()[-2]], [max_mass_ts1, max_mass_ts2])
#     VOC_end = VOC_end[PTRMS[key].keys()[-2]].mean()

#     # Determine SOA yield and consumed VOC
#     Y_SOA, Delta_VOC = SOA_yield(VOC_start_conc, VOC_end, max_particle, Mw, T)
#     print(f'{row[1]['date']} Y({key.split('_')[1]}) = {Y_SOA:.2f}')
#     print(f'VOC consumed = {Delta_VOC:.2f} ppb')

#     # Plot VOC timeseries and decay
#     fig, ax = plt.subplots(2, 1, figsize = (6.3, 6.3))

#     t_zero = pd.to_datetime(f'{row[1]['date']} {row[1]['UV on']}')
#     plot_total(ax[0], PTRMS[key], PTRMS[key].keys()[-2], None, t_zero)

#     # UV on
#     ax[0].axvspan(0, (pd.to_datetime(f'{row[1]['date']} {row[1]['UV off']}') -  t_zero) / pd.Timedelta(minutes = 1), color = 'yellow', alpha = 0.15, lw = 0, label = 'UV on')

#     # d9-BuOH inj
#     ax[0].axvspan((pd.to_datetime(f'{row[1]['date']} {row[1]['d9-BuOH inj start']}') -  t_zero) / pd.Timedelta(minutes = 1), (pd.to_datetime(f'{row[1]['date']} {row[1]['d9-BuOH inj end']}') -  t_zero) / pd.Timedelta(minutes = 1), color = 'blue', alpha = 0.15, lw = 0, label = 'd9-BuOH inj')

#     # H2O2 inj
#     ax[0].axvspan((pd.to_datetime(f'{row[1]['date']} {row[1]['H2O2 inj start']}') -  t_zero) / pd.Timedelta(minutes = 1), (pd.to_datetime(f'{row[1]['date']} {row[1]['H2O2 inj end']}') -  t_zero) / pd.Timedelta(minutes = 1), color = 'orange', alpha = 0.15, label = 'H$_{2}$O$_{2}$ inj')

#     # VOC inj
#     ax[0].axvspan((pd.to_datetime(f'{row[1]['date']} {row[1]['VOC inj start']}') -  t_zero) / pd.Timedelta(minutes = 1), (pd.to_datetime(f'{row[1]['date']} {row[1]['VOC inj end']}') -  t_zero) / pd.Timedelta(minutes = 1), color = 'green', alpha = 0.15, lw = 0, label = 'VOC inj')

#     # Determine VOC decay rate (from UV on until max particle mass is reached)
#     Fit_df = time_filtered_conc(PTRMS[key], [PTRMS[key].keys()[-2]], 
#                                 [t_zero, pd.to_datetime(f'{row[1]['date']} {row[1]['max particle conc']}')]).dropna()
#     Fit_df['Time'] = (Fit_df['Time'] - t_zero) / pd.Timedelta(minutes = 1)
#     values, errors, ndof, squares, R2 = linear_fit(Fit_df['Time'], np.log(Fit_df[PTRMS[key].keys()[-2]]), linear,
#                                                    a = -0.01, b = 5)
#     fit = linear(Fit_df['Time'], *values)

#     # Plot fitted decay
#     ax[1].scatter(Fit_df['Time'], np.log(Fit_df[PTRMS[key].keys()[-2]]), s = 10)
#     ax[1].plot(Fit_df['Time'], fit, color = 'k', lw = 1.2)
#     ax[1].text(0.05, 0.075, f'ln[{PTRMS[key].keys()[-2]}] = ({values[0]*10**(3):.2f}E-03)t + {values[1]:.3f}', 
#                transform = ax[1].transAxes, bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))

#     # print fitted decay rate
#     print(f'{row[1]['date']} ln[{key.split('_')[1]}] = ({values[0]*10**(3):.2f}E-03+-{errors[0]*10**(3):.2f}E-03)t + ({values[1]:.2f}+-{errors[1]:.2f})')

#     # Set legend, ax-labels and title
#     ax[0].legend(frameon = True)
#     ax[0].set(ylabel = 'VOC conc. (ppb)', title = f'{row[1]['date']} {key.split('_')[1]}')
#     ax[1].set(ylabel = f'ln[{key.split('_')[1]}]', xlabel = 'Time (min)')

#     fig.tight_layout()
#     fig.savefig(f'{savepath}{row[1]['date']}_{key.split('_')[1]}.jpg', dpi = 600)

#%%
def calc_D9BuOH_decay(df, d9_key, averaging_time, timestamps):

    k_D9BuOH = 2.7*10**(-12)    # cm3 molecules-1 s-1

    # Filter the timeperiod that you want to fit
    if averaging_time == None:
        filtered_df = time_filtered_conc(df, [d9_key], timestamps)
    else:
        filtered_df, _ = running_mean(df, [d9_key], 'Time', averaging_time, timestamps)
        filtered_df['Time'] = filtered_df.index
        filtered_df = filtered_df.reset_index(drop = True)

    filtered_df = filtered_df.dropna()

    # Convert timestamps into time passed in seconds
    filtered_df['time_seconds'] = (pd.to_datetime(filtered_df['Time']) - pd.to_datetime(timestamps[0])) / pd.Timedelta(seconds = 1)

    # Calculate ln(I/I0)
    I0 = filtered_df[d9_key][0]
    filtered_df['ln(I/I0)'] = np.log(filtered_df[d9_key] / I0)

    # Linear fit 
    fit_result, fit_error, filtered_df['y_fit'], filtered_df['y_error'], R2 = least_squares_reg(filtered_df['time_seconds'], filtered_df['ln(I/I0)'], linear, True)

    # filtered_df['y_fit'] = y_fit
    # filtered_df['y_error'] = y_error

    # OH concentration calculation
    # Define variables
    OH, I = symbols('OH, I')
    dOH, dI = symbols('sigma_OH, sigma_I')
    # Perimeter:
    # Define relation
    OH = (-I) / k_D9BuOH
    # Calculate uncertainty
    dOH = sqrt((OH.diff(I)*dI)**2)
    # Turn expression into numerical functions
    fOH = lambdify((I), OH)
    fdOH = lambdify((I,dI), dOH)
    # define values and their errors
    vI, vdI = fit_result[0], fit_error[0]
    # numerically evaluate expression
    vOH = fOH(vI)
    vdOH = fdOH(vI, vdI)

    # Print results
    print(f'R2 = {R2:.2f}')
    print(f'slope = ({fit_result[0]:.3e}+-{np.sqrt(fit_error[0]):.3e}) s-1')
    print(f'[OH] = ({vOH:.3e}+-{vdOH:.3e}) molecules cm-3')
    
    return filtered_df, vOH, vdOH, fit_result, fit_error

def plot_D9BuOh_decay(df, d9_key, averaging_time, timestamps):
    filtered_df, vOH, vdOH, fit_result, fit_error = calc_D9BuOH_decay(df, d9_key, averaging_time, timestamps)

    time_minutes = (pd.to_datetime(filtered_df['Time']) - pd.to_datetime(timestamps[0])) / pd.Timedelta(minutes = 1)

    fig, ax = plt.subplots(figsize = (5, 3.5))

    ax.fill_between(time_minutes, filtered_df['y_fit'] + filtered_df['y_error'], 
                    filtered_df['y_fit'] - filtered_df['y_error'],
                        alpha = 0.25, label = 'Error on fit', color = 'k', lw = 0)
    ax.plot(time_minutes, filtered_df['y_fit'], lw = 1, 
            label = ('Fit\n' f'OH = {vOH:.2e} ${{\pm}}$ {vdOH:.1e} molecules cm$^{{-3}}$'), color = 'k')
    ax.scatter(time_minutes, filtered_df['ln(I/I0)'], s = 10, label = f'Measured data', color = 'r')

    ax.legend(frameon = True)

    ax.set(xlabel = 'Time (min)', ylabel = r'ln$\frac{I}{I_{0}}$')

    return fig, ax

for key, row in zip(exp_keys, data['Timestamps'].iterrows()):
    if 'Toluene' in key:
        fig, ax = plot_D9BuOh_decay(PTRMS[key], 'm94.074 (C[12]6C[13]1H[1]9) (Conc)', None, 
                                    [f'{row[1]['date']} {row[1]['UV on']}', 
                                     f'{row[1]['date']} {row[1]['max particle conc']}']) 
    else:
        d9_key = [k for k in PTRMS[key].keys() if 'm67.129' in k]
        fig, ax = plot_D9BuOh_decay(PTRMS[key], d9_key[0], None,
                                    [f'{row[1]['date']} {row[1]['UV on']}', 
                                     f'{row[1]['date']} {row[1]['max particle conc']}'])

    ax.set(title = f'{row[1]['date']} {key.split('_')[1]}')
    
    fig.tight_layout()
    fig.savefig(f'{savepath}{row[1]['date']}_{key.split('_')[1]}_D9-BuOH.jpg', dpi = 600)
