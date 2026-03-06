#%%
import sys
sys.path.append('../../')
from Functions import *
from calculations import running_mean
plt.style.use('Style.mplstyle')
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # suppress warnings
#%%
# Import data
path = '../../../../Courses/2026 - INAR winter school/Data/'
MION = read_csv(path, '', 'Time', '%d-%b-%Y %H:%M:%S')
MION = MION['MION_Ambient_WinterSchool']
PSM24 = read_csv(f'{path}PSM/2024/', '', 'ScanTime', '%d-%b-%Y %H:%M:%S')
PSM25 = read_csv(f'{path}PSM/2025/', '', 'ScanTime', '%d-%b-%Y %H:%M:%S')
DMPS_temp = read_csv(f'{path}DMPS/', '', 'Date', '%Y-%m-%d %H:%M:%S')
DMPS = pd.merge(DMPS_temp['DMPS_TZS_2024'], DMPS_temp['Particle number conc_TZS_2024'], on = 'Time', how = 'outer').drop(['Date_x', 'Date_y'], axis = 1)
NAIS = read_csv(f'{path}NAIS/', '', 'time', '%Y-%m-%d %H:%M:%S')
for key in NAIS.keys():
    NAIS[key] = NAIS[key].drop(['time'], axis = 1)

# Merge PSM dataframes
PSM_2024_01 = pd.DataFrame()
PSM_2024 = pd.DataFrame()
for key in PSM24.keys():
    if PSM24[key].iloc[-1]['Time'] < pd.to_datetime('2024-01-23 00:00:00'):
        PSM_2024_01 = pd.concat([PSM_2024_01, PSM24[key]], ignore_index = True)
    if pd.to_datetime('2024-01-23 00:00:00') < PSM24[key].iloc[-1]['Time']:
        PSM_2024 = pd.concat([PSM_2024, PSM24[key]], ignore_index = True)
PSM_2025 = pd.DataFrame()
for key in PSM25.keys():
    PSM_2025 = pd.concat([PSM_2025, PSM25[key]], ignore_index = True)
PSM = {'2024-01': PSM_2024_01, '2024': PSM_2024, '2025': PSM_2025}
bins = [[1.17, 1.3, 1.5, 1.7, 2.5, 3.0, 3.92],
        [1.1, 1.3, 1.5, 1.7, 2.5, 3.0],
        [1.17, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0]]

# 1-hour running mean of data
PSM_running = {}
for key, binedges in zip(PSM.keys(), bins):
    PSM[key]['Total conc'] = calc_total(PSM[key], PSM[key].keys()[1:-2], binedges)
    temp = running_mean(PSM[key], PSM[key].keys()[1:-3].to_list()+['Total conc'], 'Time', '60min', 60, None)
    temp['Time'] = temp.index
    temp = temp.reset_index(drop = True)
    PSM_running[key] = temp

NAIS_running = {}
for key in NAIS.keys():
    temp = running_mean(NAIS[key], NAIS[key].keys()[:-1], 'Time', '60min', 60, ['2023-12-31 23:59:00', '2026-01-01 00:01:00'])
    temp['Time'] = temp.index + pd.Timedelta(hours = 2)
    NAIS_running[key] = temp.reset_index(drop = True)
NAIS_mask = NAIS_running['formation_rate_2_2p3_neg']['N2-2.3,-'] < 80
NAIS_running['formation_rate_2_2p3_neg'] = NAIS_running['formation_rate_2_2p3_neg'][NAIS_mask]

MION_running = running_mean(MION, MION.keys()[1:], 'Time', '60min', 60, None)
MION_running['Time'] = MION_running.index
MION_running = MION_running.reset_index(drop = True)
#%%
# Plot yearly PSM
for key, binedges in zip(PSM_running.keys(), bins):
    fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
    plot_timeseries(fig, ax, PSM_running[key], PSM_running[key].keys()[:-2], binedges, 'number', True, 'Total conc', None, '%Y/%m')
    ax[0].set(title = key)
    fig.tight_layout()
    fig.savefig(f'Figures/PSM_{key}.jpg', dpi = 600)

# Plot yearly DMPS
bin_means = []
for key in DMPS.keys()[:-2]:
    bin_means.append(float(key))
bins_list = calc_bin_edges(np.array(bin_means)) / 10**(-9)
fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
plot_timeseries(fig, ax, DMPS, DMPS.keys()[:-2], bins_list, 'number', True, 'TZS (PNC)', None, '%Y/%m')
fig.tight_layout()
fig.savefig('Figures/DMPS_2024.jpg')

# Plot yearly NAIS total concentration
fig, ax = plt.subplots(figsize = (6.3, 4))
plot_multi_total(ax, NAIS_running['tvarminne_conc_utc'], ['tot 2.5-7', 'tot 7-20'], ['2.5-7 nm', '7-20 nm'], '%Y/%m')
ax.set(title = '2024-2025')
fig.tight_layout()
fig.savefig('Figures/NAIS_tot_2024-2025.jpg', dpi = 600)

# Plot yearly NAIS neg particle concentration
fig, ax = plt.subplots(figsize = (6.3, 4))
plot_multi_total(ax, NAIS_running['tvarminne_conc_utc'], ['neg 0.8-2', 'neg 2-7', 'neg 7-20'], ['<2 nm','2-7 nm', '7-20 nm'], '%Y/%m')
ax.set(title = '2024-2025')
fig.tight_layout()
fig.savefig('Figures/NAIS_neg_2024-2025.jpg', dpi = 600)

# Plot yearly NAIS particle formation rate
fig, ax = plt.subplots(figsize = (6.3, 4))
ax, ax2 = plot_total_twinx(ax, NAIS_running['formation_rate_2_2p3_neg'], ['J2-2.3,-/N<2,-', 'N2-2.3,-'], '%Y/%m', ['J$_{2-2.3 nm}$/N$_{<2 nm}$', 'dN/dlogDp (# cm$^{-3}$)'], None)
ax.set(title = '2024-2025')
fig.tight_layout()
fig.savefig('Figures/NAIS_FR_2024-2025.jpg', dpi = 600)

# Plot yearly MION sulfate cluster
fig, ax = plt.subplots(figsize = (6.3, 4))
plot_multi_total(ax, MION_running, MION_running.keys()[2:5], ['HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)$_{2}$HSO$_{4}^{-}$'], '%Y/%m')
ax.set(title = '2024-2025', ylabel = 'ions/s')
fig.tight_layout()
fig.savefig('Figures/MION_sulfate-cluster_2024-2025.jpg', dpi = 600)
#%%
# Plot monthly PSM
for key, binedges in zip(PSM_running.keys(), bins):
    Months = [str(i).split('-')[1] for i in PSM_running[key]['Time']]
    PSM_running[key]['Month'] = Months
    for month, group in PSM_running[key].groupby('Month'):
        fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
        plot_timeseries(fig, ax, group, PSM_running[key].keys()[:-3], binedges, 'number', True, 'Total conc', None, '%m/%d')
        year = str(PSM_running[key].iloc[0]['Time']).split('-')[0]
        ax[0].set(title = f'{year}-{month}')
        fig.tight_layout()
        if key is not '2024-01':
            fig.savefig(f'Figures/PSM/Monthly/PSM_{year}-{month}.jpg', dpi = 600)
    PSM_running[key] = PSM_running[key].drop(['Month'], axis = 1)

# Plot monthly DMPS
Months = [str(i).split('-')[1] for i in DMPS['Time']]
DMPS['Month'] = Months
for month, group in DMPS.groupby('Month'):
    fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
    plot_timeseries(fig, ax, group, DMPS.keys()[:-3], bins_list, 'number', True, 'TZS (PNC)', None, '%m/%d')
    year = str(DMPS.iloc[0]['Time']).split('-')[0]
    ax[0].set(title = f'{year}-{month}')
    fig.tight_layout()
    fig.savefig(f'Figures/DMPS/Monthly/DMPS_{year}-{month}.jpg', dpi = 600)
DMPS = DMPS.drop(['Month'], axis = 1)
#%%
# Plot DMPS diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_DMPS, axes = plot_diurnal_mean(axes, DMPS, 'TZS (PNC)', 'Total conc. (# cm$^{-3}$)')
fig.tight_layout()
fig.savefig('Figures/DMPS/Diurnal_mean.jpg', dpi = 600)

# Plot PSM 2024 diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_PSM_2024, axes = plot_diurnal_mean(axes, PSM_running['2024'], 'Total conc', 'Total conc. (# cm$^{-3}$)')
fig.tight_layout()
fig.savefig('Figures/PSM/Diurnal_mean_2024.jpg', dpi = 600)

# Plot PSM 2025 diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_PSM_2025, axes = plot_diurnal_mean(axes, PSM_running['2025'], 'Total conc', 'Total conc. (# cm$^{-3}$)')
fig.tight_layout()
fig.savefig('Figures/PSM/Diurnal_mean_2025.jpg', dpi = 600)

# Plot NAIS diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_NAIS_FR, axes = plot_diurnal_mean(axes, NAIS_running['formation_rate_2_2p3_neg'], 'J2-2.3,-/N<2,-', 'J$_{2-2.3 nm}$/N$_{<2 nm}$')
fig.tight_layout()
fig.savefig('Figures/NAIS/Diurnal_mean_FR.jpg', dpi = 600)
#%%
# Plot daily DMPS
Dates = [str(i).split(' ')[0] for i in DMPS['Time']]
DMPS['Date'] = Dates
for date, group in DMPS.groupby('Date'):
    if pd.to_datetime('2024-07-12') < pd.to_datetime(date) < pd.to_datetime('2024-07-30'):
        pass
    else:
        fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
        plot_timeseries(fig, ax, group, DMPS.keys()[:-4], bins_list, 'number', True, 'TZS (PNC)', None, '%H:%M')
        ax[0].set(title = date)
        fig.tight_layout()
        fig.savefig(f'Figures/DMPS/Daily/{date.split('-')[0]}-{date.split('-')[1]}/DMPS_{date}.jpg', dpi = 600)
DMPS = DMPS.drop(['Date'], axis = 1)
#%%
# Plot 2024 daily NAIS 
merged_NAIS = pd.merge(NAIS_running['tvarminne_conc_utc'], NAIS_running['formation_rate_2_2p3_neg'], on = 'Time', how = 'outer')
Dates = [str(i).split(' ')[0] for i in merged_NAIS['Time']]
merged_NAIS['Date'] = Dates
for date, group in merged_NAIS.groupby('Date'):
    if pd.to_datetime('2024-12-31') < pd.to_datetime(date):
        pass
    else:
        fig, ax = plt.subplots(figsize = (6.3, 4))
        plot_total_twinx(ax, group, ['J2-2.3,-/N<2,-', 'N2-2.3,-'], '%H:%M', ['J$_{2-2.3 nm}$/N$_{<2 nm}$', 'dN/dlogDp (# cm$^{-3}$)'], None)
        ax.set(title = date)
        fig.tight_layout()
        fig.savefig(f'Figures/NAIS/Daily/2024/{date.split('-')[0]}-{date.split('-')[1]}/NAIS_{date}.jpg', dpi = 600)
merged_NAIS = merged_NAIS.drop(['Date'], axis = 1)
#%%
# Plot monthly NAIS MION correlation
Particle_formation = pd.merge(NAIS_running['formation_rate_2_2p3_neg'], MION_running, on = 'Time', how = 'outer')
Particle_formation['Month'] = [str(i).split('-')[1] for i in Particle_formation['Time']]
Particle_formation['Year'] = [str(i).split('-')[0] for i in Particle_formation['Time']]

labels = ['HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)$_{2}$HSO$_{4}^{-}$']
PF_keys = MION_running.keys()[2:5].to_list() + ['J2-2.3,-/N<2,-']
for year, year_group in Particle_formation.groupby('Year'):
    for month, group in year_group.groupby('Month'):
        fig = plt.figure(figsize = (9, 6.3))
        axes = [plt.subplot(2, 1, 1), plt.subplot(2, 3, 4), plt.subplot(2, 3, 5), plt.subplot(2, 3, 6)]
        ax1, ax_twin = plot_correlation_tseries(axes, group, PF_keys, '%m/%d', ['ions/s', 'J$_{2-2.3 nm}$/N$_{<2 nm}$'], labels)
        ax1.set(title = f'{year}-{month}')
        ax1.legend(labels =labels, ncols = 3)
        fig.tight_layout()
        fig.savefig(f'Figures/NAIS/NAISvsMION/{year}-{month}_NAISvsMION.jpg', dpi = 600)

#%%
# Plot daily PSM
# for key, binedges in zip(PSM.keys(), bins):
#     Dates = [str(i).split(' ')[0] for i in PSM[key]['Time']]
#     PSM[key]['Date'] = Dates
#     for date, group in PSM[key].groupby('Date'):
#         fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
#         plot_timeseries(fig, ax, group, PSM[key].keys()[1:-4], binedges, 'number', True, 'Total conc', None, '%H:%M')
#         ax[0].set(title = date)
#         fig.tight_layout()
#         fig.savefig(f'Figures/PSM/Dayly/{date.split('-')[0]}/{date.split('-')[0]}-{date.split('-')[1]}/PSM_{date}.jpg', dpi = 600)
#     PSM[key] = PSM[key].drop(['Date'], axis = 1)