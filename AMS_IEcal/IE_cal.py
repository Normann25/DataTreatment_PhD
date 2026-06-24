#%%
import sys
sys.path.append('../')
from read_data_functions import *
from plot_functions import *
from calculations import *
plt.style.use('../Style.mplstyle')
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # suppress warnings
#%%
path = '../../../Data/2026/20260624_IE-cal/'

AMS = import_data(f'{path}AMS/', '', 't_series', '%d-%m-%Y %H:%M:%S', 0)
AMS = AMS['NO3_counts']

CPC = import_data(f'{path}CPC/', '', 'ï»¿Date-Time', '%Y-%m-%d %H:%M:%S', 0)
CPC = CPC['2026-06-24 095202_1 Hz']

timestamps = [['2026-06-24 10:13:32', '2026-06-24 10:18:00'],
              ['2026-06-24 10:21:31', '2026-06-24 10:26:00'],
              ['2026-06-24 10:28:01', '2026-06-24 10:32:31'],
              ['2026-06-24 10:34:31', '2026-06-24 10:39:00'],
              ['2026-06-24 10:41:00', '2026-06-24 10:45:31'],
              ['2026-06-24 10:49:31', '2026-06-24 10:54:01'],
              ['2026-06-24 10:55:30', '2026-06-24 11:00:00']]
#%%
filtered_AMS = pd.DataFrame(columns = ['Time', 'HRNO3_M_all', 'flowrate'])
filtered_CPC = pd.DataFrame(columns = ['Time', 'Concentration (#/cm3)'])

for time in timestamps:
    temp_ams = time_filtered_conc(AMS, ['HRNO3_M_all', 'flowrate'], time)
    filtered_AMS = pd.concat([filtered_AMS, temp_ams], ignore_index = True)
    temp_cpc = running_mean(CPC, ['Concentration (#/cm3)'], 'Time', '30s', time)
    temp_cpc['Time'] = temp_cpc.index
    temp_cpc = temp_cpc.reset_index(drop = True)
    filtered_CPC = pd.concat([filtered_CPC, temp_cpc], ignore_index = True)

MPP = ((np.array(filtered_CPC['Concentration (#/cm3)']) * (np.pi/6) * 0.8 * 300**(3)*10**(-21) * 1.72 * 0.775) / 62) * 6.022*10**(23) * np.array(filtered_AMS['flowrate'])

fig, ax = plt.subplots()
fit_params, fit_errors, squares, ndof, R2 = instrument_comparison(ax, MPP, np.array(filtered_AMS['HRNO3_M_all']), 'CPC vs AMS', 
                                                                  ['CPC NO$_{3}$ signal \n (molecules/s)', 'AMS NO$_{3}$ signal \n (HZ)'], True)
ax.text(0.01, 0.75, 'f(x) = 2.94*10$^{-7}$x \n R2 = 0.98', transform = ax.transAxes)

fig.tight_layout()
fig.savefig('../../../Figures/IE_cal/20260624.jpg', dpi = 600)