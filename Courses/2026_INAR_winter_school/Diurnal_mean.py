#%%
import sys
sys.path.append('../../')
from Functions import *
plt.style.use('Style.mplstyle')
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # suppress warnings
#%%
# Import data
path = '../../../../Courses/2026 - INAR winter school/Data/'
data = read_csv(f'{path}1H-avg/', '', 'Time', '%Y-%m-%d %H:%M:%S')

DMPS_bin_means = []
for key in data['DMPS_2024'].keys()[:-2]:
    DMPS_bin_means.append(float(key))
DMPS_bins = calc_bin_edges(np.array(DMPS_bin_means)) / 10**(-9)

temp = read_csv(path, '', 'Time', '%Y-%m-%d %H:%M:%S')
TZS_data = time_filtered_conc(temp['combined_Tvarminne_v1'], temp['combined_Tvarminne_v1'].keys()[1:], ['2023-12-31 23:59', '2025-12-31 23:59'])

cluster_keys = ['HSO4-', '(H2SO4)HSO4-', '(H2SO4)2HSO4-', 'IO3-', '(HIO3)HSO4-']
aerosol_keys = ['J2-2.3-/N<2-', 'num_con_10_25nm', 'num_con_25_100nm', 'num_con_100nm']

Marine_mask = TZS_data['Marine'] == 1
Land_mask = TZS_data['Land'] == 1

comparison_dict = {'TZS Marine': TZS_data[Marine_mask], 'TZS Land': TZS_data[Land_mask]}
colors = ['tab:blue', 'red']
#%%
# Plot DMPS diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_DMPS, axes = plot_diurnal_mean(axes, data['DMPS_2024'], 'TZS (PNC)', 'Total conc. (# cm$^{-3}$)')
fig.tight_layout()
fig.savefig('../../../../Courses/2026 - INAR winter school/Figures/DMPS/Diurnal_mean.jpg', dpi = 600)

# Plot NAIS diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_NAIS_FR, axes = plot_diurnal_mean(axes, TZS_data, 'J2-2.3-/N<2-', 'J$_{2-2.3 nm}$/N$_{<2 nm}$')
fig.tight_layout()
fig.savefig('../../../../Courses/2026 - INAR winter school/Figures/NAIS/Diurnal_mean_FR.jpg', dpi = 600)
#%%
def plot_multiparameter_diurnal(axes, dictionary, df_keys, colors, season, ylabels):
    for color, dict_key in zip(colors, dictionary.keys()):
        for j, ax, in enumerate(axes):
            diurnal_df = calc_diurnal_mean(dictionary[dict_key], df_keys[j])

            diurnal_keys = [k for k in diurnal_df.keys() if season in k]
            ax.fill_between(diurnal_df['Time'], diurnal_df[diurnal_keys[2]], diurnal_df[diurnal_keys[1]], alpha = 0.2, color = color, linewidth=0)
            ax.plot(diurnal_df['Time'], diurnal_df[diurnal_keys[0]], color = color, lw = 1.2, marker = '.', label = dict_key)
            # ax.scatter(diurnal_df['Time'], diurnal_df[diurnal_keys[0]], color = color, s = 10)

            ax.set(xlabel='Time of day (h)', ylabel = ylabels[j])
        
    return axes

cluster_ylabels = ['HSO$_{4}^{-}$ (ions s$^{-1}$)', '(H$_{2}$SO$_{4}$)HSO$_{4}^{-}$ (ions s$^{-1}$)', '(H$_{2}$SO$_{4}$)$_{2}$HSO$_{4}^{-}$ (ions s$^{-1}$)', 
                   'IO$_{3}^{-}$ (ions s$^{-1}$)', '(HIO$_{3}$)HSO$_{4}^{-}$ (ions s$^{-1}$)']
aerosol_ylabels = ['J$_{2-2.3 nm}$/N$_{<2 nm}$ (s$^{-1}$)', 'N$_{10-25 nm}$ (# cm$^{-3}$)', 'N$_{25-100 nm}$ (# cm$^{-3}$)', 'N$_{>100 nm}$ (# cm$^{-3}$)']

for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    fig = plt.figure(figsize = (10, 6.3))
    axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
            plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
    plot_multiparameter_diurnal(axes, comparison_dict, cluster_keys, colors, season, cluster_ylabels)
    sublabels = ['a)', 'b)', 'c)', 'd)', 'e)']
    for ax, l in zip(axes, sublabels):
        ax.text(-0.2, 1.05, l, transform = ax.transAxes, fontsize = 12)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles = handles, labels = labels)
    fig.tight_layout()
    fig.savefig(f'../../../../Courses/2026 - INAR winter school/Figures/Diurnal variations/{season}_clusters_diurnal.jpg', dpi = 600)

    fig2, axes2 = plt.subplots(2, 2, figsize = (6.3, 6.3))
    plot_multiparameter_diurnal(axes2.flatten(), comparison_dict, aerosol_keys, colors, season, aerosol_ylabels)
    for ax in axes2.flatten():
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles = handles, labels = labels)
    fig2.tight_layout()
    fig2.savefig(f'../../../../Courses/2026 - INAR winter school/Figures/Diurnal variations/{season}_aerosols_diurnal.jpg', dpi = 600)
