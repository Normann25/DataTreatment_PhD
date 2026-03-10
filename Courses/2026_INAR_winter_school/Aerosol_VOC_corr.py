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
Marine_mask = TZS_data['Marine'] == 1
Land_mask = TZS_data['Land'] == 1

comparison_dict = {'HYDE': HYDE_data, 'TZS Marine': TZS_data[Marine_mask], 'TZS Land': TZS_data[Land_mask]}
colors = ['orange', 'mediumblue', 'green']
ax_labels = ['J$_{2-2.3 nm}$/N$_{<2 nm}$ (s$^{-1}$)', 'Monomers (molecules cm$^{-3}$)', 'N monomers (molecules cm$^{-3}$)', 
             'Dimers (molecules cm$^{-3}$)', 'N dimers (molecules cm$^{-3}$)', 'Total org. (molecules cm$^{-3}$)', 
             'DMS (ppb)', 'Isoprene (ppb)', 'Monoterpenes (ppb)']
comparison_dict = plot_seasonal_scatter(comparison_dict, ['HYDE', 'TZS Marine', 'TZS Land'], ['J2-2.3-/N<2-']+OVOC_keys, 
                      ['2024-01-01 00:00', '2025-12-31 23:59'], colors, ax_labels, 'Day', 'Figures/Correlations/')