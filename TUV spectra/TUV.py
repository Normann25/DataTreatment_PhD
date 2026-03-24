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
path = '../../../Data/2026/TUV data/'
data = import_data(path, '', None, None, None)

equator_keys = [['250621_equator_0km', '250621_equator_05km', '250621_equator_1km', '250621_equator_5km', '250621_equator_10km'], 
                ['251221_equator_0km', '251221_equator_05km', '251221_equator_1km', '251221_equator_5km', '251221_equator_10km']]
arktis_keys = [['250621_arktis_0km', '250621_arktis_05km', '250621_arktis_1km', '250621_arktis_5km', '250621_arktis_10km'], 
               ['251221_arktis_0km', '251221_arktis_05km', '251221_arktis_1km', '251221_arktis_5km', '251221_arktis_10km']]
marselis_keys = [['250621_marselis_0km', '250621_marselis_05km', '250621_marselis_1km', '250621_marselis_5km', '250621_marselis_10km'], 
                 ['251221_marselis_0km', '251221_marselis_05km', '251221_marselis_1km', '251221_marselis_5km', '251221_marselis_10km']]
dates = ['Jun 21st 2025', 'Dec 21st 2025']
altitudes = ['0 km', '0.5 km', '1 km', '5 km', '10 km']
#%%
fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
for i, keys in enumerate(equator_keys):
    n_lines = len(keys)+1
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, n_lines))
    for j, key in enumerate(keys):
        wavelength = (data[key]['LOWER WVL'] + data[key]['UPPER WVL']) / 2
        ax[i].plot(wavelength, data[key]['TOTAL'], label = altitudes[j], color = colors[j], lw = 1)

    ax[i].legend()
    ax[i].set(title = dates[i], xlabel = 'Wavelength (nm)', ylabel = 'Actinic flux (photon s$^{-1}$cm$^{-2}$nm$^{-1}$)', yscale = 'log')

# fig.suptitle('Equator')
fig.tight_layout()
fig.savefig('Solar_spectrum_equator.jpg', dpi = 600)

fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
for i, keys in enumerate(arktis_keys):
    n_lines = len(keys)+1
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, n_lines))
    for j, key in enumerate(keys):
        wavelength = (data[key]['LOWER WVL'] + data[key]['UPPER WVL']) / 2
        ax[i].plot(wavelength, data[key]['TOTAL'], label = altitudes[j], color = colors[j], lw = 1)

    ax[i].legend()
    ax[i].set(title = dates[i], xlabel = 'Wavelength (nm)', ylabel = 'Actinic flux (photon s$^{-1}$cm$^{-2}$nm$^{-1}$)')

# fig.suptitle('Arctic')
fig.tight_layout()
fig.savefig('Solar_spectrum_arctic.jpg', dpi = 600)

fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
for i, keys in enumerate(marselis_keys):
    n_lines = len(keys)+1
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, n_lines))
    for j, key in enumerate(keys):
        wavelength = (data[key]['LOWER WVL'] + data[key]['UPPER WVL']) / 2
        ax[i].plot(wavelength, data[key]['TOTAL'], label = altitudes[j], color = colors[j], lw = 1)

    ax[i].legend()
    ax[i].set(title = dates[i], xlabel = 'Wavelength (nm)', ylabel = 'Actinic flux (photon s$^{-1}$cm$^{-2}$nm$^{-1}$)')

# fig.suptitle('Marselis Forrest')
fig.tight_layout()
fig.savefig('Solar_spectrum_marselis.jpg', dpi = 600)