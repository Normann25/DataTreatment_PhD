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
paths = ['260427_Vanillin70ppb_UV_RH70/', '260428_Vanillin70ppb_UV_RH70/', '260429_Vanillin70ppb_UV_RH85/']

timestamps = [['2026-04-27 10:36', '2026-04-27 17:28'],
              ['2026-04-28 11:07', '2026-04-28 17:38'],
              ['2026-04-29 08:52', '2026-04-29 16:45']]
t_zero = ['2026-04-27 12:16', '2026-04-28 12:35', '2026-04-29 11:45']
t_UV_off = ['2026-04-28 16:36', '2026-04-29 15:45']

SMPS = {}
PTRMS = {}
for t, path in zip(t_zero, paths):
    temp_SMPS = import_SMPS(f'{parent_path}{path}SMPS/', '', 0)
    for key in temp_SMPS.keys():
        temp_SMPS[key].loc[temp_SMPS[key]['Time'] < pd.to_datetime(t), ['Median (nm)', 'Mean (nm)', 'Geo. Mean (nm)', 'Mode (nm)']] = np.nan
        SMPS[key] = temp_SMPS[key]
    if path != paths[0]:
        temp_PTR = import_PTRMS(f'{parent_path}{path}PTRMS/', '')
        for key in temp_PTR.keys():
            mask = (0 < temp_PTR[key][temp_PTR[key].keys()[5]]) & (temp_PTR[key][temp_PTR[key].keys()[5]] < 150)
            temp = temp_PTR[key][mask]
            PTRMS[key] = temp

save_path = 'Figures/2604_vanillin+UV_RH70/'

for key in SMPS.keys():
    SMPS[key].rename(columns = {SMPS[key].columns[38]:'Total concentration'}, inplace = True)
    SMPS[key] = SMPS[key].fillna(0)
#%%
SMPS_keys = [['260427_vanillin+UV_RH70_number', '260428_vanillin+UV_RH70_number', '260429_vanillin+UV_RH85_number'],
             ['260427_vanillin+UV_RH70_mass', '260428_vanillin+UV_RH70_mass', '260429_vanillin+UV_RH85_mass']
             ]
ax, ax_2 = plot_SMPS(SMPS, SMPS_keys, SMPS['260428_vanillin+UV_RH70_mass'].columns[42:-1], 'number and mass', 
                     timestamps, 10, 'Total concentration', t_zero, 1, 3, save_path)
#%%
colors = ['green', 'purple']
PTRMS_keys = ['260428_VL+UV_70RH_initial', '260429_VL+UV_RH85_initial']
for i, key in enumerate(PTRMS_keys):
    fig = plt.figure(figsize = (6.3,6.3))
    axes = [plt.subplot(2, 1, 1), plt.subplot(2, 2, 3), plt.subplot(2, 2, 4)]

    for j, conc in enumerate(PTRMS[key].keys()[4:6]):
        plot_total(axes[0], PTRMS[key], conc, colors[j], t_zero[i])
    axes[0].legend(['C$_{7}$H$_{9}$O$_{2}^{+}$', 'C$_{8}$H$_{9}$O$_{3}^{+}$'])

    UV_on = time_filtered_conc(PTRMS[key], [PTRMS[key].keys()[5]], [t_zero[i+1], t_UV_off[i]])
    UV_on['Time'] = (UV_on['Time'] - pd.to_datetime(t_zero[i+1])) / pd.Timedelta(minutes = 1)
    axes[1].scatter(UV_on['Time'], UV_on[PTRMS[key].keys()[5]], color = 'purple', s = 10)
    on_values, on_errors, on_ndof, on_squares, on_R2 = linear_fit(UV_on['Time'], UV_on[PTRMS[key].keys()[5]], linear, a = -10, b = 100)
    on_fit = linear(UV_on['Time'], *on_values)
    axes[1].plot(UV_on['Time'], on_fit, color = 'k', lw = 1.2, ls = '--')
    axes[1].text(0.05, 0.05, f'f(x) = {on_values[0]:.3f}x + {on_values[1]:.3f}', transform = axes[1].transAxes, bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))

    UV_off = time_filtered_conc(PTRMS[key], [PTRMS[key].keys()[5]], [t_UV_off[i], timestamps[i+1][1]])
    UV_off['Time'] = (UV_off['Time'] - pd.to_datetime(t_zero[i+1])) / pd.Timedelta(minutes = 1)
    axes[2].scatter(UV_off['Time'], UV_off[PTRMS[key].keys()[5]], color = 'purple', s = 10)
    off_values, off_errors, off_ndof, off_squares, off_R2 = linear_fit(UV_off['Time'], UV_off[PTRMS[key].keys()[5]], linear, a = -10, b = 100)
    off_fit = linear(UV_off['Time'], *off_values)
    axes[2].plot(UV_off['Time'], off_fit, color = 'k', lw = 1.2, ls = '--')
    axes[2].text(0.05, 0.05, f'f(x) = {off_values[0]:.3f}x + {off_values[1]:.3f}', transform = axes[2].transAxes, bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))

    for ax in axes:
        ax.set(ylabel = 'Concentration (ppb)', xlabel = 'Time (min)')
    axes[0].set_title(f'{t_zero[i+1].split(' ')[0]}')
    fig.tight_layout()
    fig.savefig(f'{save_path}{t_zero[i+1]}_PTRMS_initial.jpg', dpi = 600)

