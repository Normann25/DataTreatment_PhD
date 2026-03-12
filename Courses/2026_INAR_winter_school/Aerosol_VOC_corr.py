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

OVOC_keys = ['Monomers', 'NitrogenMonomers', 'Dimers', 'NitrogenDimers', 'TotalOrganics', 'DMSppb', 'IPppb', 'MTppb']
cluster_keys = ['HSO4-', '(H2SO4)HSO4-', '(H2SO4)2HSO4-', 'IO3-', '(HIO3)HSO4-']
PFR_vs_cluster_keys = []
normPFR_vs_cluster_keys = []
PFR_vs_normCluster_keys = []
normPFR_vs_normCluster_keys = []
for key in cluster_keys:
    TZS_data[f'J2-2.3-_vs_{key}'] = TZS_data['J2-2.3-'] / TZS_data[key]
    PFR_vs_cluster_keys.append(f'J2-2.3-_vs_{key}')
    TZS_data[f'normJ2-2.3-_vs_{key}'] = TZS_data['J2-2.3-/N<2-'] / TZS_data[key]
    normPFR_vs_cluster_keys.append(f'normJ2-2.3-_vs_{key}')
    norm_cluster = pd.to_numeric(TZS_data[key]) / pd.to_numeric(TZS_data['TotalIons'])
    TZS_data[f'J2-2.3-_vs_norm{key}'] = TZS_data['J2-2.3-'] / norm_cluster
    PFR_vs_normCluster_keys.append(f'J2-2.3-_vs_norm{key}')
    TZS_data[f'normJ2-2.3-_vs_norm{key}'] = TZS_data['J2-2.3-/N<2-'] / norm_cluster
    normPFR_vs_normCluster_keys.append(f'normJ2-2.3-_vs_norm{key}')
Marine_mask = TZS_data['Marine'] == 1
Land_mask = TZS_data['Land'] == 1

comparison_dict = {'HYDE': HYDE_data, 'TZS Marine': TZS_data[Marine_mask], 'TZS Land': TZS_data[Land_mask]}
colors = ['orange', 'mediumblue', 'red']
#%%
def plot_seasonal_scatter(data_dict, dict_keys, df_keys, timestamps, colors, ax_labels, time_of_day, x_text, y_text, xlim, ylim, save_path):
    new_dict = {}
    for key in dict_keys:
        df = data_dict[key]
        df['Hour'] = [str(i).split(':')[0] for i in df['Time']]
        df['Hour'] = [int(i.split(' ')[1]) for i in df['Hour']]
        if time_of_day == 'Day':
            hour_mask = (8 < df['Hour']) & (df['Hour'] < 16)
            new_df = df[hour_mask]
        if time_of_day == 'Night':
            hour_mask1 = 20 < df['Hour']
            hour_mask2 = 4 > df['Hour']
            new_df = pd.concat([df[hour_mask1], df[hour_mask2]])
        df.drop(['Hour'], axis = 1)
        new_dict[key] = split_season(new_df, df_keys, timestamps)

    for i, key in enumerate(df_keys[1:]):
        fig, axes = plt.subplots(2, 2, figsize = (7.5, 6.3))

        season_names = ['Winter', 'Spring', 'Summer', 'Autumn']
        for ax, season in zip(axes.flatten(), season_names):
            pvalues = []
            for j, dict_key in enumerate(dict_keys):
                mask = new_dict[dict_key]['Season'] == season
                temp = new_dict[dict_key][mask]
                ax.scatter(temp[key], temp[df_keys[0]], color = colors[j], s = 5)
                ax.set(xscale = 'log', yscale = 'log', title = season,
                    xlabel = ax_labels[i+1], ylabel = ax_labels[0])
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                statistic, pvalue = stats.spearmanr(temp.dropna()[key], temp.dropna()[df_keys[0]])
                pvalues.append(statistic)
            textobjs = []
            for j, s in enumerate(pvalues):
                rho = r'$\rho$'
                s = f'{rho} = {s:.3f}'
                text = ax.text(x_text, y_text+j*0.1, s, c=colors[j], transform=ax.transAxes)
                        # bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
                textobjs.append(text)
            print(textobjs)        
            fig.canvas.draw()
            xmin = min([t.get_window_extent().xmin for t in textobjs])
            xmax = max([t.get_window_extent().xmax for t in textobjs])
            ymin = min([t.get_window_extent().ymin for t in textobjs])
            ymax = max([t.get_window_extent().ymax for t in textobjs])
        #     xmin, ymin = fig.transFigure.inverted().transform((xmin, ymin))
        #     xmax, ymax = fig.transFigure.inverted().transform((xmax, ymax))
            rect = patches.Rectangle((x_text, y_text),xmax-x_text,ymax-y_text, facecolor='white', alpha=0.9, transform=ax.transAxes)
            ax.add_patch(rect)

        axes[0][1].legend(labels = dict_keys, bbox_to_anchor = (1, 1, 0, 0))
        fig.tight_layout()
        if '/' in df_keys[0]:
            name = f'{df_keys[0].split('/')[0]}{df_keys[0].split('/')[1].split('<')[0]}'
            fig.savefig(f'{save_path}{time_of_day}_{name}_vs_{key}_corr.jpg', dpi = 600)
        else:
            fig.savefig(f'{save_path}{time_of_day}_{df_keys[0]}_vs_{key}_corr.jpg', dpi = 600)

    return new_dict

# Plot NAIS vs OVOC correlations
ax_labels_org = ['J$_{2-2.3 nm}$/N$_{<2 nm}$ (# s$^{-1}$)', 'Monomers (# cm$^{-3}$)', 'N monomers (# cm$^{-3}$)', 
             'Dimers (# cm$^{-3}$)', 'N dimers (# cm$^{-3}$)', 'Total org. (# cm$^{-3}$)', 
             'DMS (ppb)', 'Isoprene (ppb)', 'Monoterpenes (ppb)']
seasonal_corr_org = plot_seasonal_scatter(comparison_dict, ['HYDE', 'TZS Marine', 'TZS Land'], ['J2-2.3-/N<2-']+OVOC_keys, 
                      ['2024-01-01 00:00', '2025-12-31 23:59'], colors, ax_labels_org, 'Day', 0.05, 0.07, None, (10**(-7), 3*10**(-3)), 'Figures/Correlations/')

# Plot VOC vs OVOC correlation
for label, key in zip(ax_labels_org[1:6], OVOC_keys[:5]):
    DMS_vs_OVOC = plot_seasonal_scatter(comparison_dict, ['HYDE', 'TZS Marine', 'TZS Land'], [key]+[OVOC_keys[5]], 
                                ['2024-01-01 00:00', '2025-12-31 23:59'], colors, [label, ax_labels_org[6]], 'Day', 0.05, 0.07, None, None, 'Figures/Correlations/')
    Monoterpenes_vs_OVOC = plot_seasonal_scatter(comparison_dict, ['HYDE', 'TZS Marine', 'TZS Land'], [key]+[OVOC_keys[7]], 
                                ['2024-01-01 00:00', '2025-12-31 23:59'], colors, [label, ax_labels_org[8]], 'Day', 0.62, 0.07, None, None, 'Figures/Correlations/')

# Plot NAIS vs cluster correlation
ax_labels_cluster = ['J$_{2-2.3 nm}$/N$_{<2 nm}$ (# s$^{-1}$)', 'HSO$_{4}^{-}$ (ions s$^{-1}$)', '(H$_{2}$SO$_{4}$)HSO$_{4}^{-}$ (ions s$^{-1}$)', 
                     '(H$_{2}$SO$_{4}$)$_{2}$HSO$_{4}^{-}$ (ions s$^{-1}$)', 'IO$_{3}^{-}$ (ions s$^{-1}$)', '(HIO$_{3}$)HSO$_{4}^{-}$ (ions s$^{-1}$)']
seasonal_corr_cluser1 = plot_seasonal_scatter(comparison_dict, ['TZS Marine', 'TZS Land'], ['J2-2.3-/N<2-']+cluster_keys, 
                      ['2024-01-01 00:00', '2025-12-31 23:59'], colors[1:], ax_labels_cluster, 'Day', 0.05, 0.07, None, (10**(-7), 3*10**(-3)), 'Figures/Correlations/')
seasonal_corr_cluser2 = plot_seasonal_scatter(comparison_dict, ['TZS Marine', 'TZS Land'], ['J2-2.3-']+cluster_keys, 
                      ['2024-01-01 00:00', '2025-12-31 23:59'], colors[1:], ['J$_{2-2.3 nm}$'] + ax_labels_cluster[1:], 'Day', 0.05, 0.07, None, None, 'Figures/Correlations/')
#%%
# Temperature vs cluster weighted formation rate (full year)
ax_labels = ['Temperature (C)', 'J$_{2-2.3 nm}$/HSO$_{4}^{-}$', 'J$_{2-2.3 nm}$/(H$_{2}$SO$_{4}$)HSO$_{4}^{-}$', 'J$_{2-2.3 nm}$/(H$_{2}$SO$_{4}$)$_{2}$HSO$_{4}^{-}$',
             'J$_{2-2.3 nm}$/IO$_{3}^{-}$', 'J$_{2-2.3 nm}$/(HIO$_{3}$)HSO$_{4}^{-}$']
fig = plt.figure(figsize = (10, 6.3))
axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
        plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
for i, key in enumerate(PFR_vs_cluster_keys):
    for j, dict_key in enumerate(['TZS Marine', 'TZS Land']):
        temp = time_filtered_conc(comparison_dict[dict_key], ['avg_T_C', key], ['2024-01-01 00:00', '2025-12-31 23:59'])   
        plot_correlation([axes[i]], temp, ['avg_T_C', key], colors[j+1], [ax_labels[i+1], 'Temperature (C)'], 'Day')
        statistic, pvalue = stats.spearmanr(temp.dropna()['avg_T_C'], temp.dropna()[key])
        rho = r'$\rho$'
        s = f'{rho} = {statistic:.3f}'
        axes[i].text(0.05, 0.05+j*0.1, s, c=colors[j+1], transform=axes[i].transAxes,
                bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
        axes[i].set(yscale = 'log')
fig.tight_layout()
fig.savefig('Figures/Correlations/Temp_vs_weighted-PF.jpg', dpi = 600)

# Temperature vs cluster weighted formation rate (summer)
fig = plt.figure(figsize = (10, 6.3))
axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
        plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
for i, key in enumerate(PFR_vs_cluster_keys):
    for j, dict_key in enumerate(['TZS Marine', 'TZS Land']):
        temp = split_season(comparison_dict[dict_key], ['avg_T_C', key], ['2024-01-01 00:00', '2025-12-31 23:59'])
        summer_mask = temp['Season'] == 'Summer'
        plot_correlation([axes[i]], temp[summer_mask], ['avg_T_C', key], colors[j+1], [ax_labels[i+1], 'Temperature (C)'], 'Day')
        statistic, pvalue = stats.spearmanr(temp[summer_mask].dropna()['avg_T_C'], temp[summer_mask].dropna()[key])
        rho = r'$\rho$'
        s = f'{rho} = {statistic:.3f}'
        axes[i].text(0.05, 0.05+j*0.1, s, c=colors[j+1], transform=axes[i].transAxes,
                bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
        axes[i].set(yscale = 'log')
fig.tight_layout()
fig.savefig('Figures/Correlations/Summer_Temp_vs_weighted-PF.jpg', dpi = 600)

#%%
ax_labels = ['Temperature (C)', r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{HSO_{4}^{-}}$', 
             r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{(H_{2}SO_{4})HSO_{4}^{-}}$', r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{(H_{2}SO_{4})_{2}HSO_{4}^{-}}$',
             r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{IO_{3}^{-}}$', r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{(HIO_{3})HSO_{4}^{-}}$']
fig = plt.figure(figsize = (10, 6.3))
axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
        plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
for i, key in enumerate(normPFR_vs_cluster_keys):
    for j, dict_key in enumerate(['TZS Marine', 'TZS Land']):
        temp = time_filtered_conc(comparison_dict[dict_key], ['avg_T_C', key], ['2024-01-01 00:00', '2025-12-31 23:59'])   
        plot_correlation([axes[i]], temp, ['avg_T_C', key], colors[j+1], [ax_labels[i+1], 'Temperature (C)'], 'Day')
        statistic, pvalue = stats.spearmanr(temp.dropna()['avg_T_C'], temp.dropna()[key])
        rho = r'$\rho$'
        s = f'{rho} = {statistic:.3f}'
        axes[i].text(0.05, 0.05+j*0.1, s, c=colors[j+1], transform=axes[i].transAxes,
                bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
        axes[i].set(yscale = 'log')
fig.tight_layout()
fig.savefig('Figures/Correlations/Temp_vs_ClusterWeighted-normPF.jpg', dpi = 600)

# Temperature vs cluster weighted formation rate (summer)
fig = plt.figure(figsize = (10, 6.3))
axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
        plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
for i, key in enumerate(normPFR_vs_cluster_keys):
    for j, dict_key in enumerate(['TZS Marine', 'TZS Land']):
        temp = split_season(comparison_dict[dict_key], ['avg_T_C', key], ['2024-01-01 00:00', '2025-12-31 23:59'])
        summer_mask = temp['Season'] == 'Summer'
        plot_correlation([axes[i]], temp[summer_mask], ['avg_T_C', key], colors[j+1], [ax_labels[i+1], 'Temperature (C)'], 'Day')
        statistic, pvalue = stats.spearmanr(temp[summer_mask].dropna()['avg_T_C'], temp[summer_mask].dropna()[key])
        rho = r'$\rho$'
        s = f'{rho} = {statistic:.3f}'
        axes[i].text(0.05, 0.05+j*0.1, s, c=colors[j+1], transform=axes[i].transAxes,
                bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
        axes[i].set(yscale = 'log')
fig.tight_layout()
fig.savefig('Figures/Correlations/Summer_Temp_vs_ClusterWeighted-normPF.jpg', dpi = 600)
#%%
ax_labels = ['Temperature (C)', r'$\frac{J_{2-2.3 nm}}{HSO_{4}^{-}/Total ions}$', 
             r'$\frac{J_{2-2.3 nm}}{(H_{2}SO_{4})HSO_{4}^{-}/Total ions}$', r'$\frac{J_{2-2.3 nm}}{(H_{2}SO_{4})_{2}HSO_{4}^{-}/Total ions}$',
             r'$\frac{J_{2-2.3 nm}}{IO_{3}^{-}/Total ions}$', r'$\frac{J_{2-2.3 nm}}{(HIO_{3})HSO_{4}^{-}/Total ions}$']
fig = plt.figure(figsize = (10, 6.3))
axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
        plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
for i, key in enumerate(PFR_vs_normCluster_keys):
    for j, dict_key in enumerate(['TZS Marine', 'TZS Land']):
        temp = time_filtered_conc(comparison_dict[dict_key], ['avg_T_C', key], ['2024-01-01 00:00', '2025-12-31 23:59'])   
        plot_correlation([axes[i]], temp, ['avg_T_C', key], colors[j+1], [ax_labels[i+1], 'Temperature (C)'], 'Day')
        statistic, pvalue = stats.spearmanr(temp.dropna()['avg_T_C'], temp.dropna()[key])
        rho = r'$\rho$'
        s = f'{rho} = {statistic:.3f}'
        axes[i].text(0.05, 0.05+j*0.1, s, c=colors[j+1], transform=axes[i].transAxes,
                bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
        axes[i].set(yscale = 'log')
fig.tight_layout()
fig.savefig('Figures/Correlations/Temp_vs_normClusterWeighted-PF.jpg', dpi = 600)

# Temperature vs cluster weighted formation rate (summer)
fig = plt.figure(figsize = (10, 6.3))
axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
        plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
for i, key in enumerate(normPFR_vs_cluster_keys):
    for j, dict_key in enumerate(['TZS Marine', 'TZS Land']):
        temp = split_season(comparison_dict[dict_key], ['avg_T_C', key], ['2024-01-01 00:00', '2025-12-31 23:59'])
        summer_mask = temp['Season'] == 'Summer'
        plot_correlation([axes[i]], temp[summer_mask], ['avg_T_C', key], colors[j+1], [ax_labels[i+1], 'Temperature (C)'], 'Day')
        statistic, pvalue = stats.spearmanr(temp[summer_mask].dropna()['avg_T_C'], temp[summer_mask].dropna()[key])
        rho = r'$\rho$'
        s = f'{rho} = {statistic:.3f}'
        axes[i].text(0.05, 0.05+j*0.1, s, c=colors[j+1], transform=axes[i].transAxes,
                bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
        axes[i].set(yscale = 'log')
fig.tight_layout()
fig.savefig('Figures/Correlations/Summer_Temp_vs_normClusterWeighted-PF.jpg', dpi = 600)
#%%
ax_labels = ['Temperature (C)', r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{HSO_{4}^{-}/Total ions}$', 
             r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{(H_{2}SO_{4})HSO_{4}^{-}/Total ions}$', r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{(H_{2}SO_{4})_{2}HSO_{4}^{-}/Total ions}$',
             r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{IO_{3}^{-}/Total ions}$', r'$\frac{J_{2-2.3 nm}/N_{<2 nm}}{(HIO_{3})HSO_{4}^{-}/Total ions}$']
fig = plt.figure(figsize = (10, 6.3))
axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
        plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
for i, key in enumerate(normPFR_vs_normCluster_keys):
    for j, dict_key in enumerate(['TZS Marine', 'TZS Land']):
        temp = time_filtered_conc(comparison_dict[dict_key], ['avg_T_C', key], ['2024-01-01 00:00', '2025-12-31 23:59'])   
        plot_correlation([axes[i]], temp, ['avg_T_C', key], colors[j+1], [ax_labels[i+1], 'Temperature (C)'], 'Day')
        statistic, pvalue = stats.spearmanr(temp.dropna()['avg_T_C'], temp.dropna()[key])
        rho = r'$\rho$'
        s = f'{rho} = {statistic:.3f}'
        axes[i].text(0.05, 0.05+j*0.1, s, c=colors[j+1], transform=axes[i].transAxes,
                bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
        axes[i].set(yscale = 'log')
fig.tight_layout()
fig.savefig('Figures/Correlations/Temp_vs_normClusterWeighted-normPF.jpg', dpi = 600)

# Temperature vs cluster weighted formation rate (summer)
fig = plt.figure(figsize = (10, 6.3))
axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
        plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
for i, key in enumerate(normPFR_vs_normCluster_keys):
    for j, dict_key in enumerate(['TZS Marine', 'TZS Land']):
        temp = split_season(comparison_dict[dict_key], ['avg_T_C', key], ['2024-01-01 00:00', '2025-12-31 23:59'])
        summer_mask = temp['Season'] == 'Summer'
        plot_correlation([axes[i]], temp[summer_mask], ['avg_T_C', key], colors[j+1], [ax_labels[i+1], 'Temperature (C)'], 'Day')
        statistic, pvalue = stats.spearmanr(temp[summer_mask].dropna()['avg_T_C'], temp[summer_mask].dropna()[key])
        rho = r'$\rho$'
        s = f'{rho} = {statistic:.3f}'
        axes[i].text(0.05, 0.05+j*0.1, s, c=colors[j+1], transform=axes[i].transAxes,
                bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
        axes[i].set(yscale = 'log')
fig.tight_layout()
fig.savefig('Figures/Correlations/Summer_Temp_vs_normClusterWeighted-normPF.jpg', dpi = 600)
#%%
ax_labels = ['Temperature (C)', 'HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)HSO$_{4}^{-}$', '(H$_{2}$SO$_{4}$)$_{2}$HSO$_{4}^{-}$',
             'IO$_{3}^{-}$', '(HIO$_{3}$)HSO$_{4}^{-}$']
fig = plt.figure(figsize = (10, 6.3))
axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2), plt.subplot(2, 3, 3),
        plt.subplot(2, 3, 4), plt.subplot(2, 3, 5)]
ylimits = [(10**(-4), 4), (10**(-4), 7), (10**(-4), 5), (10**(-4), 2), (10**(-4), 0.8)]
for i, key in enumerate(cluster_keys):
    for j, dict_key in enumerate(['TZS Marine', 'TZS Land']):
        temp = time_filtered_conc(comparison_dict[dict_key], ['avg_T_C', key], ['2024-01-01 00:00', '2025-12-31 23:59'])   
        plot_correlation([axes[i]], temp, ['avg_T_C', key], colors[j+1], [ax_labels[i+1], 'Temperature (C)'], 'Day')
        statistic, pvalue = stats.spearmanr(temp.dropna()['avg_T_C'], temp.dropna()[key])
        rho = r'$\rho$'
        s = f'{rho} = {statistic:.3f}'
        axes[i].text(0.05, 0.8+j*0.1, s, c=colors[j+1], transform=axes[i].transAxes,
                bbox=dict(ec = 'white', fc = 'white', lw = 0.5, alpha = 0.9))
        axes[i].set(yscale = 'log', ylim = ylimits[i])
fig.tight_layout()
fig.savefig('Figures/Correlations/Temp_vs_clusters.jpg', dpi = 600)
#%%
temperature_bins = np.arange(0, 27, 3)
temperature_bin_means = (temperature_bins[:-1] + temperature_bins[1:]) / 2
TZS_data['Bins'] = pd.cut(TZS_data['avg_T_C'], temperature_bins)

binned_cluster_Marine = {}
for key in cluster_keys:
    temp = pd.DataFrame()
    for i, group in enumerate(TZS_data[Marine_mask].groupby('Bins')):
        print(group)
        temp[str(temperature_bin_means[i])] = group[1][key]
    binned_cluster_Marine[key] = temp
# binned_cluster_Marine['Temperature'] = temperature_bin_means

# binned_cluster_Land = pd.DataFrame(columns = cluster_keys)
# for bin, group in TZS_data[Land_mask].groupby('Bins'):
#     temp = pd.DataFrame()
#     for key in cluster_keys:
#         conc = group.dropna()[key].mean()
#         temp[key] = [conc]
#         temp[f'{key} 90 %'] = np.percentile(np.array(group.dropna()[key]), 90)
#         temp[f'{key} 10 %'] = np.percentile(np.array(group.dropna()[key]), 10)
#     binned_cluster_Land = pd.concat([binned_cluster_Land, temp], ignore_index=True)
# binned_cluster_Land['Temperature'] = temperature_bin_means


fig, ax = plt.subplots(figsize = (6.3, 3))
ax.boxplot(binned_cluster_Marine['HSO4-'])