import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from datetime import datetime
sys.path.append('../../')
from calculations import linear, linear_fit
from mpl_axes_aligner import align

def file_list(path, parent_path): # List files in specified folder
    ParentPath = os.path.abspath(parent_path)
    if ParentPath not in sys.path:
        sys.path.insert(0, ParentPath)
    
    files = os.listdir(path)

    return files

def format_timestamps(timestamps, old_format, new_format): # Change format of timestamps in dataframe
    new_timestamps = []
    for timestamp in timestamps:
        old_datetime = datetime.strptime(str(timestamp), old_format)
        new_datetime = old_datetime.strftime(new_format)
        new_timestamps.append(new_datetime)
    return pd.to_datetime(new_timestamps, format=new_format)

def read_NAIS(path, parent_path):
    return

def read_csv(path, parent_path, timelabel, time_format):
    data_dict = {}
    files = file_list(path, parent_path)

    for file in files:
        if file.endswith('.csv'):
            name = file.split('.')[0]
            with open(os.path.join(path, file), 'r') as f:
                df = pd.read_csv(f)
            df['Time'] = format_timestamps(df[timelabel], time_format, '%d/%m/%Y %H:%M:%S')
            data_dict[name] = df

    return data_dict

def calc_total(df, df_keys, bins):
    temp = pd.DataFrame()
    for i, key in enumerate(df_keys):
        decades = np.log10(bins[i+1]/bins[i])
        temp[key] = df[key] * decades
    
    Total_conc = np.zeros(len(df['Time']))
    for i, row in temp.iterrows():
        Total_conc[i] += np.sum(row)

    return Total_conc

def calc_bin_edges(dp):
    """
    Calculate bin edges given bin centers  
    Parameters
    ----------
    dp : pandas series of lenght n
        bin center diameters
    Returns
    -------
    array of lenght n+1 containing bin edges
    """
    dp_arr = dp
    logdp_mid = np.log10(dp_arr)
    logdp = (logdp_mid[:-1]+logdp_mid[1:])/2.0
    maxval = [logdp_mid.max()+(logdp_mid.max()-logdp.max())]
    minval = [logdp_mid.min()-(logdp.min()-logdp_mid.min())]
    logdp = np.concatenate((minval,logdp,maxval))
    
    return 10**logdp

def time_filtered_conc(df, df_keys, timestamps):
    start_time, end_time = pd.to_datetime(timestamps[0]), pd.to_datetime(timestamps[1])
    time = pd.to_datetime(df['Time'])
    time_filter = (time >= start_time) & (time <= end_time)
    filtered_time = np.array(time[time_filter])
    new_df = pd.DataFrame({'Time': filtered_time})

    for key in df_keys:
        conc = np.array(df[key])
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]

        new_df[key] = filtered_conc

    return new_df

def running_mean(df, concentration, timelabel, interval, timestamps):

    if timestamps == None:      
        df = df.set_index(timelabel)

        # Resample the data to bins 
        new_df = df[concentration].resample(interval).mean() 

    else:
        new_df = time_filtered_conc(df, concentration, timestamps)
        new_df = new_df.set_index('Time')

        # Resample the data to bins 
        new_df = new_df.resample(interval).mean() 

    return new_df

def plot_heatmap(ax, df, df_keys, time, bin_edges, normed, time_format):

    data = np.array(df[df_keys])

    if normed == False:
        dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
        data=data/dlogDp
        
    # Generate an extra time bin, which is needed for the meshgrid
    dt = time[1]-time[0]
    new_time = time - dt
    new_time = np.append(new_time, new_time[-1]+dt)

    # generate 2d meshgrid for the x, y, and z data of the 3D color plot
    y, x = np.meshgrid(bin_edges, new_time)
    
    # Fill the generated mesh with particle concentration data
    p1 = ax.pcolormesh(x, y, data, cmap='viridis', shading='flat', norm=mpl.colors.LogNorm(vmin = 1)) # ,vmin=y_min, vmax=y_max

    ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
    ax.tick_params(axis = 'x', which = 'minor', bottom = False)
    if time_format == '%Y/%m':
        ax.set_xlabel("Time (yyyy/mm)")
    if time_format == '%m/%d':
        ax.set_xlabel("Time (mm/dd)")
    if time_format == '%H:%M':
        ax.set_xlabel("Time (HH:MM)")
    plt.subplots_adjust(hspace=0.05)
        
    # Make the y-scal logarithmic and set a label
    ax.set_yscale("log")
    ax.set_ylabel("Dp (nm)")
    return ax, p1

def plot_total(ax, df, conc_key, clr, time_format):
    ax.plot(df['Time'], df[conc_key], lw = 1, color = clr)

    ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
    ax.tick_params(axis = 'x', which = 'minor', bottom = False)
    if time_format == '%Y/%m':
        ax.set_xlabel("Time (yyyy/mm)")
    if time_format == '%m/%d':
        ax.set_xlabel("Time (mm/dd)")
    if time_format == '%H:%M':
        ax.set_xlabel("Time (HH:MM)")
    plt.subplots_adjust(hspace=0.05)
    return ax

def plot_multi_total(ax, df, df_keys, labels, time_format):
    n_lines = len(df_keys)
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, n_lines))

    for key, color in zip(df_keys, colors):
        plot_total(ax, df, key, color, time_format)
    
    ax.legend(labels = labels)
    ax.set(ylabel = 'dN/dlogDp (# cm$^{-3}$)')
    return ax

def plot_total_twinx(ax, df, df_keys, time_format, ylabels, labels):
    if len(df_keys) < 3:
        plot_total(ax, df, df_keys[0], 'r', time_format)
        ax.set_ylabel(ylabels[0], color = 'r')
        ax.tick_params(axis = 'y', labelcolor='r')

        ax2 = ax.twinx()
        plot_total(ax2, df, df_keys[1], 'b', time_format)
        ax2.tick_params(axis = 'y', labelcolor='b')
        ax2.set_ylabel(ylabels[1], color = 'b')

    else:
        plot_multi_total(ax, df, df_keys[:-1], labels, time_format)
        ax.set_ylabel(ylabels[0])
        ax2 = ax.twinx()
        plot_total(ax2, df, df_keys[-1], 'b', time_format)
        ax2.tick_params(axis = 'y', labelcolor='b')
        ax2.set_ylabel(ylabels[1], color = 'b')
    return ax, ax2

def plot_correlation(axes, df, df_keys, color, ax_labels):
    df['Hour'] = [str(i).split(':')[0] for i in df['Time']]
    df['Hour'] = [int(i.split(' ')[1]) for i in df['Hour']]
    hour_mask = (8 < df['Hour']) & (df['Hour'] < 16)

    for i, ax in enumerate(axes):
        ax.scatter(df[hour_mask][df_keys[-1]], df[hour_mask][df_keys[i]], color = color, s = 10)

        # Calculate correlation
        valuesfit, errorsfit, Ndof_fit, squares_fit, R2 = linear_fit(df.dropna()[hour_mask][df_keys[-1]], df.dropna()[hour_mask][df_keys[i]], linear, a_guess = 1, b_guess = 0)
        ax.text(0.55, 0.05, f'R2 = {R2:.3f}', transform=ax.transAxes)
                #, bbox=dict(ec = 'gray', fc = 'white', lw = 0.5))

        ax.set(xlabel = ax_labels[0], ylabel = ax_labels[i+1])
    df = df.drop(['Hour'], axis = 1)

    return

def plot_correlation_tseries(axes, df, df_keys, time_format, ax_labels, labels):
    ax1, ax1_twin = plot_total_twinx(axes[0], df, df_keys, time_format, ax_labels, labels)
    # Adjust the plotting range of two y axes
    org1 = 0.0  # Origin of first axis
    org2 = 0.0  # Origin of second axis
    pos = 0.05  # Position the two origins are aligned
    align.yaxes(ax1, org1, ax1_twin, org2, pos)

    plot_correlation(axes[1:], df, df_keys, 'indigo', [ax_labels[1]]+labels)

    return ax1, ax1_twin

def plot_timeseries(fig, ax, df, df_keys, bin_edges, datatype, normed, total, timestamps, time_format):

    if datatype == 'number and mass':
        df_number, df_mass = df[0], df[1]

        if total is not None:
            if timestamps is not None:
                df_number, df_mass = time_filtered_conc(df_number, df_keys+[total], timestamps), time_filtered_conc(df_mass, df_keys+[total], timestamps)
            ax1, p1 = plot_heatmap(ax[0][0], df_number, df_keys, np.array(df_number['Time']), bin_edges, normed, time_format)
            ax2, p2 = plot_heatmap(ax[0][1], df_mass, df_keys, np.array(df_mass['Time']), bin_edges, normed, time_format)

            ax3 = plot_total(ax[1], df_number, total, 'r', time_format)
  
            ax4 = plot_total(ax[1], df_mass, total, 'r', time_format)

            ax3.set_ylabel('Total number conc. (# cm$^{-3}$)')
            ax4.set_ylabel('Total mass conc. ($\mu$g m$^{-3}$)')

        else:
            if timestamps is not None:
                df_number, df_mass = time_filtered_conc(df_number, df_keys, timestamps), time_filtered_conc(df_mass, df_keys, timestamps)
            ax1, p1 = plot_heatmap(ax[0], df_number, df_keys, np.array(df_number['Time']), bin_edges, normed, time_format)
            ax2, p2 = plot_heatmap(ax[1], df_mass, df_keys, np.array(df_mass['Time']), bin_edges, normed, time_format)

        # Insert coloarbar and label it
        col1 = fig.colorbar(p1, ax=ax1)
        col2 = fig.colorbar(p2, ax=ax2)

        col1.set_label('dN/dlogDp (# cm$^{-3}$)')
        col2.set_label('dM/dlogDp ($\mu$g m$^{-3}$)')

    else:
        if total is not None:
            if timestamps is not None:
                df = time_filtered_conc(df, df_keys+[total], timestamps)
            ax1, p1 = plot_heatmap(ax[0], df, df_keys, np.array(df['Time']), bin_edges, normed, time_format)        
            ax2 = plot_total(ax[1], df, total, 'r', time_format)

        else:
            if timestamps is not None:
                df = time_filtered_conc(df, df_keys, timestamps)
            ax1, p1 = plot_heatmap(ax, df, df_keys, np.array(df['Time']), bin_edges, normed, time_format)

        # Insert coloarbar and label it
        col = fig.colorbar(p1, ax=ax1)
        if datatype == "number":
            col.set_label('dN/dlogDp (# cm$^{-3}$)')
            if total != None:
                ax2.set_ylabel('Total conc. (# cm$^{-3}$)')
        elif datatype == "mass":
            col.set_label('dM/dlogDp ($\mu$g m$^{-3}$)')
            if total != None:
                ax2.set_ylabel('Total conc. ($\mu$g m$^{-3}$)')

def calc_diurnal_mean(df, conc_key):
    new_df = df.dropna()
    new_df['Month'] = [str(i).split('-')[1] for i in new_df['Time']]
    new_df['Date'] = [str(i).split(' ')[0] for i in new_df['Time']]

    month_of_the_year = [['12', '01', '02'],
                         ['03', '04', '05'],
                         ['06', '07', '08'],
                         ['09', '10', '11']]
    season_names = ['Winter', 'Spring', 'Summer', 'Fall']
    diurnal_df = pd.DataFrame({'Time': np.arange(0, 24, 1)})

    for months_of_season, season in zip(month_of_the_year, season_names):
        temp = pd.DataFrame(columns = np.arange(0, 24, 1))
        for month, month_group in new_df.groupby('Month'):        
            if month in months_of_season:
                for date, date_group in month_group.groupby('Date'):
                    try:
                        date_temp = pd.DataFrame([date_group[conc_key].values], columns = np.arange(0, 24, 1))
                        temp = pd.concat([temp, date_temp], ignore_index = True)
                    except ValueError:
                        pass         

        mean = []
        percentile_90 = []
        percentile_10 = []
        for temp_key in temp.keys():
            mean.append(temp[temp_key].mean())
            percentile_90.append(np.percentile(np.array(temp[temp_key]), 90))
            percentile_10.append(np.percentile(np.array(temp[temp_key]), 10))
        diurnal_df[f'{season} mean'] = mean
        diurnal_df[f'{season} 90%'] = percentile_90
        diurnal_df[f'{season} 10%'] = percentile_10
        
    return diurnal_df

def plot_diurnal_mean(axes, df, conc_key, ylabel):
    diurnal_df = calc_diurnal_mean(df, conc_key)

    for ax, key in zip(axes.flatten(), diurnal_df.keys()[1::3]):

        ax.fill_between(diurnal_df['Time'], diurnal_df[f'{key.split(' ')[0]} 10%'], diurnal_df[f'{key.split(' ')[0]} 90%'], alpha = 0.2, color = 'tab:blue', linewidth=0)
        ax.plot(diurnal_df['Time'], diurnal_df[key], color = 'tab:blue', lw = 1.2)
        ax.scatter(diurnal_df['Time'], diurnal_df[key], color = 'tab:blue', s = 10)

        ax.set(xlabel='Time of day (h)', ylabel = ylabel, title = f'{key.split(' ')[0]}')
    
    return diurnal_df, axes