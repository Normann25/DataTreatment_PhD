import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from datetime import datetime
import linecache
import xarray as xr

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
        if file.endswith('.csv') or file.endswith('.CSV'):
            separations = [',', ';']
            name = file.split('.')[0]
            for sep in separations:
                try:
                    with open(os.path.join(path, file), 'r') as f:
                        df = pd.read_csv(f, sep = sep)
                    df['Time'] = format_timestamps(df[timelabel], time_format, '%d/%m/%Y %H:%M:%S')
                    data_dict[name] = df

                except KeyError:
                    pass
                except ValueError:
                    pass

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
    
    # Set the upper and/or lower limit of the color scale based on input
    y_min = np.nanmin(data)
    # y_max = np.nanmax(data)
    
    # Fill the generated mesh with particle concentration data
    if y_min < 1:
        y_min = 1
    p1 = ax.pcolormesh(x, y, data, cmap='viridis', shading='flat', norm=mpl.colors.LogNorm(vmin = y_min)) # ,vmin=y_min, vmax=y_max

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
    ax.set(ylabel = 'dN/dlogDp (# cm$^{-3}$)', yscale = 'log')
    return

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