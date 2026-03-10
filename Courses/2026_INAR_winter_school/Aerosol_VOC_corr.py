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
TZS_data = data['combined_Tvarminne_v1']
data['HYDE_formation_rate_2_2p3_neg'] = running_mean(data['HYDE_formation_rate_2_2p3_neg'], data['HYDE_formation_rate_2_2p3_neg'].keys()[1:], 'Time', '60min', None)
HYDE_data = pd.merge(data['HYDE_formation_rate_2_2p3_neg'], data['VOCs_HYDE_hourly_averages_ppb'], on = 'Time', how = 'outer')
HYDE_data = pd.merge(HYDE_data, data['HYDE_OVOC_clean'], on = 'Time', how = 'outer')
#%%
# NAIS OVOC correlations
OVOC_keys = ['Monomers', 'NitrogenMonomers', 'Dimers', 'NitrogenDimers', 'TotalOrganics']
Marine_mask = TZS_data['Marine'] == 1
Land_mask = TZS_data['Land'] == 1

def split_season(df):
    new_df = df
    new_df['Month'] = [str(i).split('-')[1] for i in new_df['Time']]
    new_df['Date'] = [str(i).split(' ')[0] for i in new_df['Time']]

    months_of_year = [['12', '01', '02'],
                      ['03', '04', '05'],
                      ['06', '07', '08'],
                      ['09', '10', '11']]
    season_names = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_df = pd.DataFrame()
    for months_of_season, season in zip(months_of_year, season_names):
        temp = pd.DataFrame()
        for month, group in df.groupby('Month'):
            if month in months_of_season:
                temp = pd.concat([temp, group.drop(['Month'], axis = 1)], ignore_index = True)
        temp['Season'] = [season]*len(temp['Time'])
        seasonal_df = pd.concat([seasonal_df, temp], ignore_index = True)
    return seasonal_df

def plot_seasonal_scatter(data_dict, dict_keys, df_keys, colors, y_label, time_of_day, save_path):
    new_dict = {}
    for key in data_dict.keys():
        new_dict[key] = split_season(data_dict[key])
    for key in df_keys[1:]:
        fig, axes = plt.subplots(2, 2, figsize = (6.3, 6.3))

        for i, dict_key in enumerate(dict_keys):
            for group in zip(new_dict[dict_key].groupby('Season')):
                plot_correlation(axes.flatten(), group[1], [key, df_keys[0]], colors[i], [key, y_label], time_of_day)
                ax.set(xscale = 'log', yscale = 'log', title = group[0])
        fig.tight_layout()
        fig.savefig(f'{save_path}{df_keys[0]}vs{key}_corr.jpg', dpi = 600)

    return new_dict

comparison_dict = {'HYDE': HYDE_data, 'TZS Marine': TZS_data[Marine_mask], 'TZS Land': TZS_data[Land_mask]}
colors = ['orange', 'royalblue', 'forestgreen']

plot_seasonal_scatter(comparison_dict, ['HYDE', ])