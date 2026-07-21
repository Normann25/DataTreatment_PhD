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
path = '../../../Data/2026/20260721_IE-cal/'
date = '2026-07-21'

AMS = import_data(f'{path}AMS/', '', 't_series', '%d-%m-%Y %H:%M:%S', 0)
AMS = AMS['NO3_counts']

CPC = import_data(f'{path}CPC/', '', 'ï»¿Date-Time', '%Y-%m-%d %H:%M:%S', 0)
CPC = CPC['2026-07-21 092154_1 Hz']

timestamps = [[f'{date} 09:22:30', f'{date} 09:27:01'],
              [f'{date} 09:29:00', f'{date} 09:33:31'],
              [f'{date} 09:35:30', f'{date} 09:40:02'],
              [f'{date} 09:42:00', f'{date} 09:46:31'],
              [f'{date} 09:49:30', f'{date} 09:54:01'],
              [f'{date} 09:58:30', f'{date} 10:03:03']]
#%%
filtered_AMS = pd.DataFrame(columns = ['Time', 'HRNO3_M_IECal', 'flowrate'])
filtered_CPC = pd.DataFrame(columns = ['Time', 'Concentration (#/cm3)'])

for time in timestamps:
    temp_ams = time_filtered_conc(AMS, ['HRNO3_M_IECal', 'flowrate'], time)
    filtered_AMS = pd.concat([filtered_AMS, temp_ams], ignore_index = True)
    temp_cpc = running_mean(CPC, ['Concentration (#/cm3)'], 'Time', '30s', time)
    temp_cpc['Time'] = temp_cpc.index
    temp_cpc = temp_cpc.reset_index(drop = True)
    filtered_CPC = pd.concat([filtered_CPC, temp_cpc], ignore_index = True)

MPP = ((np.array(filtered_CPC['Concentration (#/cm3)']) * (np.pi/6) * 0.8 * 300**(3)*10**(-21) * 1.72 * 0.775) / 62) * 6.022*10**(23) * np.array(filtered_AMS['flowrate'])

fig, ax = plt.subplots()
fit_params, fit_errors, squares, ndof, R2 = instrument_comparison(ax, MPP, np.array(filtered_AMS['HRNO3_M_IECal']), 'CPC vs AMS', 
                                                                  ['CPC NO$_{3}$ signal \n (molecules/s)', 'AMS NO$_{3}$ signal \n (HZ)'], True)
ax.text(0.01, 0.75, f'f(x) = {fit_params[0]*10**(7):.2f}e-07x \n R2 = {R2:.2f}', transform = ax.transAxes)

fig.tight_layout()
fig.savefig('../../../Figures/IE_cal/20260721.jpg', dpi = 600)