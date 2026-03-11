#%%
from scipy import stats
from matplotlib import transforms
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
data = read_csv(path, '', 'Time', '%Y-%m-%d %H:%M:%S')
TZS_data = data['combined_Tvarminne_v1']
data['HYDE_formation_rate_2_2p3_neg'] = running_mean(data['HYDE_formation_rate_2_2p3_neg'], data['HYDE_formation_rate_2_2p3_neg'].keys()[1:], 'Time', '60min', None)
HYDE_data = pd.merge(data['HYDE_formation_rate_2_2p3_neg'], data['VOCs_HYDE_hourly_averages_ppb'], on = 'Time', how = 'outer')
HYDE_data = pd.merge(HYDE_data, data['HYDE_OVOC_clean'], on = 'Time', how = 'outer')
#%%
# NAIS OVOC correlations
OVOC_keys = ['Monomers', 'NitrogenMonomers', 'Dimers', 'NitrogenDimers', 'TotalOrganics', 'DMSppb', 'IPppb', 'MTppb']
cluster_keys = ['HSO4-', '(H2SO4)HSO4-', '(H2SO4)2HSO4-', 'IO3-', '(HIO3)HSO4-']
PF_vs_cluster_keys = []
for key in cluster_keys:
    TZS_data[f'J2-2.3-_vs_{key}'] = TZS_data['J2-2.3-'] / TZS_data[key]
    PF_vs_cluster_keys.append(f'J2-2.3-_vs_{key}')
Marine_mask = TZS_data['Marine'] == 1
Land_mask = TZS_data['Land'] == 1

comparison_dict = {'HYDE': HYDE_data, 'TZS Marine': TZS_data[Marine_mask], 'TZS Land': TZS_data[Land_mask]}
colors = ['orange', 'mediumblue', 'green']
#%%
ax_labels = ['J$_{2-2.3 nm}$/N$_{<2 nm}$ (# s$^{-1}$)', 'Monomers (# cm$^{-3}$)', 'N monomers (# cm$^{-3}$)', 
             'Dimers (# cm$^{-3}$)', 'N dimers (# cm$^{-3}$)', 'Total org. (# cm$^{-3}$)', 
             'DMS (ppb)', 'Isoprene (ppb)', 'Monoterpenes (ppb)']
seasonal_corr = plot_seasonal_scatter(comparison_dict, ['HYDE', 'TZS Marine', 'TZS Land'], ['J2-2.3-/N<2-']+OVOC_keys, 
                      ['2024-01-01 00:00', '2025-12-31 23:59'], colors, ax_labels, 'Day', 0.05, 0.05, 'Figures/Correlations/')

for label, key in zip(ax_labels[1:6], OVOC_keys[:5]):
    Monoterpenes_vs_OVOC = plot_seasonal_scatter(comparison_dict, ['HYDE', 'TZS Marine', 'TZS Land'], [OVOC_keys[5]]+[key], 
                                ['2024-01-01 00:00', '2025-12-31 23:59'], colors, [ax_labels[6], label], 'Day', 0.62, 0.05, 'Figures/Correlations/')
    Monoterpenes_vs_OVOC = plot_seasonal_scatter(comparison_dict, ['HYDE', 'TZS Marine', 'TZS Land'], [OVOC_keys[7]]+[key], 
                                ['2024-01-01 00:00', '2025-12-31 23:59'], colors, [ax_labels[8], label], 'Day', 0.62, 0.05, 'Figures/Correlations/')
#%%
print(comparison_dict['TZS Land'].keys())
ax_labels = ['Temperature (C)', 'J$_{2-2.3 nm}$/HSO$_{4}^{-}$', 'J$_{2-2.3 nm}$/(H$_{2}$SO$_{4}$)HSO$_{4}^{-}$', 'J$_{2-2.3 nm}$/(H$_{2}$SO$_{4}$)$_{2}$HSO$_{4}^{-}$',
             'J$_{2-2.3 nm}$/IO$_{3}^{-}$', 'J$_{2-2.3 nm}$/(HIO$_{3}$)HSO$_{4}^{-}$']
T_vs_PFR = plot_seasonal_scatter(comparison_dict, ['TZS Marine', 'TZS Land'], ['avg_T_C']+PF_vs_cluster_keys, 
                                 ['2024-01-01 00:00', '2025-12-31 23:59'], colors[1:], ax_labels, 'Day', 0.05, 0.05, 'Figures/Correlations/')
