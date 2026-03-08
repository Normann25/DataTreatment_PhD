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
data = read_csv(path, '', 'Time', '%Y-%m-%d %H:%M:%S')

DMPS_bin_means = []
for key in data['DMPS_2024'].keys()[:-2]:
    DMPS_bin_means.append(float(key))
DMPS_bins = calc_bin_edges(np.array(DMPS_bin_means)) / 10**(-9)

PSM_bins = [[1.17, 1.3, 1.5, 1.7, 2.5, 3.0, 3.92],
            [1.1, 1.3, 1.5, 1.7, 2.5, 3.0],
            [1.17, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0]]
#%%
# Plot yearly PSM
for key, binedges in zip(['PSM_2024-01', 'PSM_2024', 'PSM_2025'], PSM_bins):
    fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
    plot_timeseries(fig, ax, data[key], data[key].keys()[:-2], binedges, 'number', True, 'Total conc', None, '%Y/%m')
    ax[0].set(title = key)
    fig.tight_layout()
    fig.savefig(f'Figures/PSM_{key}.jpg', dpi = 600)

# Plot yearly DMPS
fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
plot_timeseries(fig, ax, data['DMPS_2024'], data['DMPS_2024'].keys()[:-2], DMPS_bins, 'number', True, 'TZS (PNC)', None, '%Y/%m')
fig.tight_layout()
fig.savefig('Figures/DMPS_2024.jpg')

# Plot yearly NAIS total concentration
fig, ax = plt.subplots(figsize = (6.3, 4))
plot_multi_total(ax, data['NAIS_TZS_1H-avg'], ['tot 2.5-7', 'tot 7-20'], ['2.5-7 nm', '7-20 nm'], '%Y/%m')
ax.set(title = '2024-2025')
fig.tight_layout()
fig.savefig('Figures/NAIS_tot_2024-2025.jpg', dpi = 600)

# Plot yearly NAIS neg particle concentration
fig, ax = plt.subplots(figsize = (6.3, 4))
plot_multi_total(ax, data['NAIS_TZS_1H-avg'], ['neg 0.8-2', 'neg 2-7', 'neg 7-20'], ['<2 nm','2-7 nm', '7-20 nm'], '%Y/%m')
ax.set(title = '2024-2025', yscale = 'log')
fig.tight_layout()
fig.savefig('Figures/NAIS_neg_2024-2025.jpg', dpi = 600)

# Plot yearly NAIS particle formation rate
fig, ax = plt.subplots(figsize = (6.3, 4))
ax, ax2 = plot_total_twinx(ax, data['NAIS_formation_rate_neg_1H-avg'], ['J2-2.3,-/N<2,-', 'N2-2.3,-'], '%Y/%m', ['J$_{2-2.3 nm}$/N$_{<2 nm}$', 'dN/dlogDp (# cm$^{-3}$)'], None)
ax.set(title = '2024-2025')
fig.tight_layout()
fig.savefig('Figures/NAIS_FR_2024-2025.jpg', dpi = 600)

# Plot yearly MION sulfate cluster
fig, ax = plt.subplots(figsize = (6.3, 4))
plot_multi_total(ax, data['MION_TZS_1H-avg'], data['MION_TZS_1H-avg'].keys()[1:4], ['HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)$_{2}$HSO$_{4}^{-}$'], '%Y/%m')
ax.set(title = '2024-2025', ylabel = 'ions/s')
fig.tight_layout()
fig.savefig('Figures/MION_sulfate-cluster_2024-2025.jpg', dpi = 600)
#%%
# Plot monthly PSM
for key, binedges in zip(['PSM_2024-01', 'PSM_2024', 'PSM_2025'], PSM_bins):
    Months = [str(i).split('-')[1] for i in data[key]['Time']]
    data[key]['Month'] = Months
    for month, group in data[key].groupby('Month'):
        fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
        plot_timeseries(fig, ax, group, data[key].keys()[:-3], binedges, 'number', True, 'Total conc', None, '%m/%d')
        year = str(data[key].iloc[0]['Time']).split('-')[0]
        ax[0].set(title = f'{year}-{month}')
        fig.tight_layout()
        if key is not '2024-01':
            fig.savefig(f'Figures/PSM/Monthly/PSM_{year}-{month}.jpg', dpi = 600)
    data[key] = data[key].drop(['Month'], axis = 1)

# Plot monthly DMPS
Months = [str(i).split('-')[1] for i in data['DMPS_2024']['Time']]
data['DMPS_2024']['Month'] = Months
for month, group in data['DMPS_2024'].groupby('Month'):
    fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
    plot_timeseries(fig, ax, group, data['DMPS_2024'].keys()[:-3], DMPS_bins, 'number', True, 'TZS (PNC)', None, '%m/%d')
    year = str(data['DMPS_2024'].iloc[0]['Time']).split('-')[0]
    ax[0].set(title = f'{year}-{month}')
    fig.tight_layout()
    fig.savefig(f'Figures/DMPS/Monthly/DMPS_{year}-{month}.jpg', dpi = 600)
data['DMPS_2024'] = data['DMPS_2024'].drop(['Month'], axis = 1)
#%%
# Plot DMPS diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_DMPS, axes = plot_diurnal_mean(axes, data['DMPS_2024'], 'TZS (PNC)', 'Total conc. (# cm$^{-3}$)')
fig.tight_layout()
fig.savefig('Figures/DMPS/Diurnal_mean.jpg', dpi = 600)

# Plot PSM 2024 diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_PSM_2024, axes = plot_diurnal_mean(axes, data['PSM_2024'], 'Total conc', 'Total conc. (# cm$^{-3}$)')
fig.tight_layout()
fig.savefig('Figures/PSM/Diurnal_mean_2024.jpg', dpi = 600)

# Plot PSM 2025 diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_PSM_2025, axes = plot_diurnal_mean(axes, data['PSM_2025'], 'Total conc', 'Total conc. (# cm$^{-3}$)')
fig.tight_layout()
fig.savefig('Figures/PSM/Diurnal_mean_2025.jpg', dpi = 600)

# Plot NAIS diurnal mean
fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))
diurnal_NAIS_FR, axes = plot_diurnal_mean(axes, data['NAIS_formation_rate_neg_1H-avg'], 'J2-2.3,-/N<2,-', 'J$_{2-2.3 nm}$/N$_{<2 nm}$')
fig.tight_layout()
fig.savefig('Figures/NAIS/Diurnal_mean_FR.jpg', dpi = 600)
#%%
# Plot daily DMPS and NAIS 2024
temp = time_filtered_conc(data['NAIS_formation_rate_neg_1H-avg'], ['J2-2.3,-/N<2,-', 'N2-2.3,-'], ['2024-01-31 23:59', '2025-01-01 00:00'])
DMPS_NAIS = pd.merge(data['DMPS_2024'], temp, on = 'Time', how = 'outer')
Dates = [str(i).split(' ')[0] for i in DMPS_NAIS['Time']]
DMPS_NAIS['Date'] = Dates
for date, group in DMPS_NAIS.groupby('Date'):
    if pd.to_datetime('2024-07-12') < pd.to_datetime(date) < pd.to_datetime('2024-07-30'):
        pass
    else:
        fig, ax = plt.subplots(2, 1, figsize = (6.3, 6))
        plot_timeseries(fig, ax[0], group, data['DMPS_2024'].keys()[:-2], DMPS_bins, 'number', True, None, None, '%H:%M')
        ax[0].set(title = date)
        plot_total_twinx(ax[1], group, ['J2-2.3,-/N<2,-', 'N2-2.3,-'], '%H:%M', ['J$_{2-2.3 nm}$/N$_{<2 nm}$', 'dN/dlogDp (# cm$^{-3}$)'], None)
        fig.tight_layout()
        fig.savefig(f'Figures/DMPS/Daily/{date.split('-')[0]}-{date.split('-')[1]}/DMPS_{date}.jpg', dpi = 600)
DMPS_NAIS = DMPS_NAIS.drop(['Date'], axis = 1)
#%%
# Plot 2025 daily NAIS 
Dates = [str(i).split(' ')[0] for i in data['NAIS_TZS_1H-avg']['Time']]
data['NAIS_TZS_1H-avg']['Date'] = Dates
for date, group in data['NAIS_TZS_1H-avg'].groupby('Date'):
    if pd.to_datetime('2025-01-01') > pd.to_datetime(date):
        pass
    else:
        fig, ax = plt.subplots(figsize = (6.3, 3))
        plot_total_twinx(ax, group, ['J2-2.3,-/N<2,-', 'N2-2.3,-'], '%H:%M', ['J$_{2-2.3 nm}$/N$_{<2 nm}$', 'dN/dlogDp (# cm$^{-3}$)'], None)
        ax.set(title = date)
        fig.tight_layout()
        fig.savefig(f'Figures/NAIS/Daily/2025/{date.split('-')[0]}-{date.split('-')[1]}/NAIS_{date}.jpg', dpi = 600)
data['NAIS_TZS_1H-avg'] = data['NAIS_TZS_1H-avg'].drop(['Date'], axis = 1)
#%%
# Plot monthly NAIS MION correlation
Particle_formation = pd.merge(data['NAIS_formation_rate_neg_1H-avg'], data['MION_TZS_1H-avg'], how = 'outer', on = 'Time')
Particle_formation['Month'] = [str(i).split('-')[1] for i in Particle_formation['Time']]
Particle_formation['Year'] = [str(i).split('-')[0] for i in Particle_formation['Time']]

labels = ['HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)$_{2}$HSO$_{4}^{-}$']
PF_keys = data['MION_TZS_1H-avg'].keys()[1:4].to_list() + ['J2-2.3,-/N<2,-']
for year, year_group in Particle_formation.groupby('Year'):
    for month, group in year_group.groupby('Month'):
        # group_mask = group['J2-2.3,-/N<2,-'] > 0

        fig = plt.figure(figsize = (9, 6.3))
        axes = [plt.subplot(2, 1, 1), plt.subplot(2, 3, 4), plt.subplot(2, 3, 5), plt.subplot(2, 3, 6)]

        ax1, ax_twin = plot_correlation_tseries(axes, group, PF_keys, '%m/%d', ['ions/s', 'J$_{2-2.3 nm}$/N$_{<2 nm}$'], labels)
        ax1.set(title = f'{year}-{month}')
        ax1.legend(labels =labels, ncols = 3)
        fig.tight_layout()
        fig.savefig(f'Figures/NAIS/NAISvsMION/{year}-{month}_NAISvsMION.jpg', dpi = 600)
Particle_formation = Particle_formation.drop(['Month', 'Year'], axis = 1)
#%%
event_dates = [['2024-03-04 23:59', '2024-03-07 00:00'],
               ['2024-03-11 23:59', '2024-03-14 00:00'],
               ['2024-03-17 23:59', '2024-03-20 00:00'],
               ['2024-04-03 23:59', '2024-04-06 00:00'],
               ['2024-04-09 23:59', '2024-04-12 00:00'],
               ['2024-04-13 23:59', '2024-04-17 00:00'],
               ['2024-04-21 23:59', '2024-04-24 00:00'],
               ['2024-05-06 23:59', '2024-05-08 00:00'], 
               ['2024-05-11 23:59', '2024-05-13 00:00'],
               ['2024-05-15 23:59', '2024-05-18 00:00'],
               ['2024-06-08 23:59', '2024-06-11 00:00'],
               ['2024-06-20 23:59', '2024-06-25 00:00'],
               ['2024-06-28 23:59', '2024-06-30 00:00'], 
               ['2024-07-04 23:59', '2024-07-06 00:00'],
               ['2024-08-11 23:59', '2024-08-14 00:00'],
               ['2024-08-18 23:59', '2024-08-21 00:00'],
               ['2024-08-24 23:59', '2024-08-27 00:00'],
               ['2024-09-09 23:59', '2024-09-12 00:00'],
               ['2024-09-14 23:59', '2024-09-17 00:00'],
               ['2024-09-18 23:59', '2024-09-22 00:00'],
               ['2024-09-28 23:59', '2024-10-01 00:00'],
               ['2024-10-02 23:59', '2024-10-04 00:00'],
               ['2024-10-13 23:59', '2024-10-17 00:00'],
               ['2024-10-25 23:59', '2024-10-29 00:00'],
               ['2024-12-01 23:59', '2024-12-04 00:00'],
               ['2024-12-09 23:59', '2024-12-13 00:00'],
               ['2024-12-24 23:59', '2024-12-28 00:00']]

merged = pd.merge(data['NAIS_formation_rate_neg_1H-avg'], data['MION_TZS_1H-avg'], on = 'Time', how = 'outer')
merged = pd.merge(merged, data['MION_NO3_1H-avg'], on = 'Time', how = 'outer')
merged_keys = ['J2-2.3,-/N<2,-', 'HSO4-', '(H2SO4)HSO4-', '(H2SO4)2HSO4-', 'SA', 'IA', 'MSA']

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