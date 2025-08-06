import sys
sys.path.append('..')
from read_data_functions import *
from plot_functions import *
plt.style.use('Style.mplstyle')

path = 'O:/Nat_Chem-Aerosol-data/Data/Raw_Data/2025/SMPSxSASS/'
parentpath = '../../../../../../'

SMPS = import_SMPS(path, parentpath, 0)

bin_labels = SMPS['2025-08-05_093558_SMPS_Number'].keys()[42:-1]

bin_Dp = [7.04]
for label in bin_labels:
    bin_Dp.append(float(label))

timestamps = ['2025-08-05 09:48:21', '2025-08-06 08:44:39']

fig1, ax1 = plt.subplots(2,1, figsize = (6.3, 6))
plot_timeseries(fig1, ax1, SMPS['2025-08-05_093558_SMPS_Number'], bin_labels, bin_Dp, 'number', timestamps, True, 'Total Concentration (#/cm³)', None)
fig1.tight_layout()

fig2, ax2 = plt.subplots(2,1, figsize = (6.3, 6))
plot_timeseries(fig2, ax2, SMPS['2025-08-05_093558_SMPS_Mass'], bin_labels, bin_Dp, 'mass', timestamps, True, 'Total Concentration (µg/m³)', None)
fig2.tight_layout()

fig3, ax3 = plt.subplots(figsize = (3.3, 3))
mean_number, error_number, mean_mass, error_mass, ax3, ax3_2 = plot_bin_mean(ax3, timestamps, SMPS['2025-08-05_093558_SMPS_Number'], SMPS['2025-08-05_093558_SMPS_Mass'], bin_labels, 'Time', bin_Dp[1:], None, 0.10, None, True)
ax3.set_xlim(bin_Dp[1], max(bin_Dp))
fig3.tight_layout()