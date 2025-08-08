import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sys.path.append('..')
from calculations import *
#%%
def plot_inset(ax, height, loc, bb2a, plot_width, xdata, ydata, width, bar, timeseries, timestamps):
    inset_ax = inset_axes(ax,
                            width = plot_width, # width = % of parent_bbox
                            height = height, # height : 1 inch
                            loc = loc,
                            bbox_to_anchor = bb2a,
                            bbox_transform = ax.transAxes) # placement in figure
    if bar:
        artist = inset_ax.bar(xdata, ydata, width)
    
    if timeseries:
        start_time = pd.to_datetime(timestamps[0])
        end_time = pd.to_datetime(timestamps[1])
        time = pd.to_datetime(xdata)
        time_filter = (time >= start_time) & (time <= end_time)

        filtered_time = np.array(time[time_filter])
        conc = np.array(ydata)
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]
        artist = inset_ax.plot(filtered_time, filtered_conc)

        # Set the x-axis major formatter to a date format
        inset_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    inset_ax.set(xlabel = None, ylabel = None)

    return artist, inset_ax

def plot_heatmap(ax, df, df_keys, time, bin_edges, cutpoint, normed, SASS):
    
    if SASS:
        data = np.array([df['CorrectedSpectralDensity'][df['ScanNumber'] == 1].tolist()])
        for scan_id, group in df.groupby('ScanNumber'):
            if scan_id != 1:
                conc = group['CorrectedSpectralDensity'].tolist()
                if len(conc) < 599:
                    conc.append(0)                
                if (scan_id % 2) == 0:
                    conc = conc[::-1]                              
                data = np.concatenate((data, np.array([conc])), axis = 0)
            #else:
                #data = np.append(data, conc.tolist(), axis = 0)

        data = np.array(data[:max(df['ScanNumber'])])
        
        # Generate an extra time bin, which is needed for the meshgrid
        time = np.unique(time)
        dt = time[1]-time[0]
        new_time = time - pd.Timedelta(minutes = 10)
        new_time = np.append(new_time, time[-1])     

        size = df['Size'][df['ScanNumber'] == 1].tolist() + [111.9]
        
        # generate 2d meshgrid for the x, y, and z data of the 3D color plot
        y, x = np.meshgrid(np.array(size), new_time)
    
    else:
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
    y_max = np.nanmax(data)
    
    # Fill the generated mesh with particle concentration data
    p1 = ax.pcolormesh(x, y, data, cmap='jet',vmin=y_min, vmax=y_max,shading='flat')

    if cutpoint != None:
        ax.hlines(cutpoint, new_time[0], new_time[-1], colors = 'white', linestyles = '--')

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
    ax.set_xlabel("Time / HH:MM")
    plt.subplots_adjust(hspace=0.05)
        
    # Make the y-scal logarithmic and set a label
    ax.set_yscale("log")
    ax.set_ylabel("Dp / nm")
    return ax, p1

def plot_total(ax, df, conc_key, clr, SASS):
    if SASS:
        total_conc = calc_total_conc(df, [111, 5490], 202.5, ['Size', 'CorrectedSpectralDensity'])
        ax.plot(total_conc['Time'], total_conc['Total Concentration'], lw = 1, color = clr)
        
    else:
        ax.plot(df['Time'], df[conc_key], lw = 1, color = clr)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
    ax.set_xlabel("Time / HH:MM")
    plt.subplots_adjust(hspace=0.05)
    return ax

def plot_timeseries(fig, ax, df, df_keys, bin_edges, datatype, timestamps, normed, total, cutpoint, SASS):
    
    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])

    if datatype == 'number and mass':
        df_number, df_mass = df[0], df[1]

        time = pd.to_datetime(df_number['Time'])

        time_filter = (time >= start_time) & (time <= end_time)

        filtered_time = np.array(time[time_filter])

        new_df_number, new_df_mass = pd.DataFrame({'Time': filtered_time}), pd.DataFrame({'Time': filtered_time})

        for key in df_keys:
            conc_number, conc_mass = np.array(df_number[key]), np.array(df_mass[key])
            conc_number, conc_mass = pd.to_numeric(conc_number, errors='coerce'), pd.to_numeric(conc_mass, errors='coerce')
            filtered_number, filtered_mass = conc_number[time_filter], conc_mass[time_filter]

            new_df_number[key], new_df_mass[key] = filtered_number, filtered_mass

        if total != None:
            ax1, p1 = plot_heatmap(ax[0][0], new_df_number, df_keys, filtered_time, bin_edges, cutpoint, normed, SASS)
            ax2, p2 = plot_heatmap(ax[0][1], new_df_mass, df_keys, filtered_time, bin_edges, cutpoint, normed, SASS)

            total_df_number = pd.DataFrame({'Time': filtered_time})
            conc = np.array(df_number[total])
            conc = pd.to_numeric(conc, errors='coerce')
            total_df_number[total] = conc[time_filter]
            ax3 = plot_total(ax[1], total_df_number, total, 'r', SASS)

            total_df_mass = pd.DataFrame({'Time': filtered_time})
            conc = np.array(df_mass[total])
            conc = pd.to_numeric(conc, errors='coerce')
            total_df_mass[total] = conc[time_filter]     
            ax4 = plot_total(ax[1], total_df_mass, total, 'r', SASS)

            ax3.set_ylabel('Total number conc. / cm$^{-3}$')
            ax4.set_ylabel('Total mass conc. / $\mu$g m$^{-3}$')

        else:
            ax1, p1 = plot_heatmap(ax[0], new_df_number, df_keys, filtered_time, bin_edges, cutpoint, normed, SASS)
            ax2, p2 = plot_heatmap(ax[1], new_df_mass, df_keys, filtered_time, bin_edges, cutpoint, normed, SASS)

        # Insert coloarbar and label it
        col1 = fig.colorbar(p1, ax=ax1)
        col2 = fig.colorbar(p2, ax=ax2)

        col1.set_label('dN/dlogDp / cm$^{-3}$')
        col2.set_label('dM/dlogDp / $\mu$g m$^{-3}$')

    else:
        time = pd.to_datetime(df['Time'])

        time_filter = (time >= start_time) & (time <= end_time)

        filtered_time = np.array(time[time_filter])

        new_df = pd.DataFrame({'Time': filtered_time})

        for key in df_keys:
            conc = np.array(df[key])
            conc = pd.to_numeric(conc, errors='coerce')
            filtered_conc = conc[time_filter]

            new_df[key] = filtered_conc

        if total != None:
            ax1, p1 = plot_heatmap(ax[0], new_df, df_keys, filtered_time, bin_edges, cutpoint, normed, SASS)
            
            if SASS:
                ax2 = plot_total(ax[1], new_df, None, 'r', SASS)
            else:
                total_df = pd.DataFrame({'Time': filtered_time})
                conc = np.array(df[total])
                conc = pd.to_numeric(conc, errors='coerce')
                total_df[total] = conc[time_filter]
                
                ax2 = plot_total(ax[1], total_df, total, 'r', SASS)

        else:
            ax1, p1 = plot_heatmap(ax, new_df, df_keys, filtered_time, bin_edges, cutpoint, normed, SASS)

        # Insert coloarbar and label it
        col = fig.colorbar(p1, ax=ax1)
        if datatype == "number":
            col.set_label('dN/dlogDp / cm$^{-3}$')
            if total != None:
                ax2.set_ylabel('Total concentration / cm$^{-3}$')
        elif datatype == "mass":
            col.set_label('dM/dlogDp / $\mu$g m$^{-3}$')
            if total != None:
                ax2.set_ylabel('Total concentration / $\mu$g m$^{-3}$')

def plot_bin_mean(ax, timestamps, df_number, df_mass, df_keys, timelabel, bin_Dp, bin_edges, cut_point, mass):
    mean_number, std_number = bin_mean(timestamps, df_number, df_keys, timelabel)

    if bin_edges != None:   # bin_egdes should only be different from None when the dataset is not normalized
        dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
        mean_number=mean_number/dlogDp
        std_number=std_number/dlogDp

    min_std_number = [m - std for m, std in zip(mean_number, std_number)]
    max_std_number = [m + std for m, std in zip(mean_number, std_number)]

    if cut_point == None:
        ax.fill_between(bin_Dp, min_std_number, max_std_number, alpha=0.2, color='tab:blue', linewidth=0)
        ax.plot(bin_Dp, mean_number, color='tab:blue', lw = 1)
    else:
        df_number = pd.DataFrame({'Bin mean': bin_Dp, 'Concentration': mean_number, 'Std min': min_std_number, 'Std max': max_std_number})
        
        lower_cut = df_number['Bin mean'] < cut_point
        upper_cut = df_number['Bin mean'] > cut_point

        ax.fill_between(df_number['Bin mean'][lower_cut], df_number['Std min'][lower_cut], df_number['Std max'][lower_cut], alpha=0.2, color='tab:blue', linewidth=0)
        ax.plot(df_number['Bin mean'][lower_cut], df_number['Concentration'][lower_cut], color='tab:blue', lw = 1.2)

        ax.fill_between(df_number['Bin mean'][upper_cut], df_number['Std min'][upper_cut], df_number['Std max'][upper_cut], alpha=0.2, color='tab:blue', linewidth=0)
        ax.plot(df_number['Bin mean'][upper_cut], df_number['Concentration'][upper_cut], color='tab:blue', lw = 1.2)

    # Explicitly set ylabel color for primary axis
    ax.tick_params(axis = 'y', labelcolor='tab:blue')
    ax.set_ylabel('dN/dlogDp / cm$^{-3}$', color='tab:blue')

    ax.set(xlabel='Particle diameter / nm', xscale='log')

    if mass:
        mean_mass, std_mass = bin_mean(timestamps, df_mass, df_keys, timelabel)

        if bin_edges != None:
            dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
            mean_mass=mean_mass/dlogDp
            std_mass=std_mass/dlogDp

        min_std_mass = [m - std for m, std in zip(mean_mass, std_mass)]
        max_std_mass = [m + std for m, std in zip(mean_mass, std_mass)]

        # Create a secondary y-axis for mass concentration
        ax2 = ax.twinx()
        
        # Plotting for the mass concentration
        if cut_point == None:
            ax2.fill_between(bin_Dp, min_std_mass, max_std_mass, alpha=0.2, color='red', linewidth=0)
            ax2.plot(bin_Dp, mean_mass, color='red', lw = 1)
        else:
            df_mass = pd.DataFrame({'Bin mean': bin_Dp, 'Concentration': mean_mass, 'Std min': min_std_mass, 'Std max': max_std_mass})
            
            lower_cut = df_mass['Bin mean'] < cut_point
            upper_cut = df_mass['Bin mean'] > cut_point

            ax.fill_between(df_mass['Bin mean'][lower_cut], df_mass['Std min'][lower_cut], df_mass['Std max'][lower_cut], alpha=0.2, color='red', linewidth=0)
            ax.plot(df_mass['Bin mean'][lower_cut], df_mass['Concentration'][lower_cut], color='red', lw = 1.2)

            ax.fill_between(df_mass['Bin mean'][upper_cut], df_mass['Std min'][upper_cut], df_mass['Std max'][upper_cut], alpha=0.2, color='red', linewidth=0)
            ax.plot(df_mass['Bin mean'][upper_cut], df_mass['Concentration'][upper_cut], color='red', lw = 1.2)

        ax2.tick_params(axis = 'y', labelcolor='red')

        # Explicitly set ylabel color for secondary axis
        ax2.set_ylabel('dM/dlogDp / $\mu$g m$^{-3}$', color='red')  # Use axis_labels[2] for clarity
    
    else:
        ax2, mean_mass = 0, 0
    
    return mean_number, mean_mass, ax, ax2

def plot_running_sizedist(fig, ax, df, bins, bin_edges, axis_labels, run_length, SASS):

    if SASS:
        n_lines = len(df['ScanNumber'].unique())
        cmap = mpl.colormaps['plasma_r']
        colors = cmap(np.linspace(0, 1, n_lines))
        
        for scan_id, group in df.groupby('ScanNumber'):
            ax.scatter(group['Size'], group['CorrectedSpectralDensity'], color = colors[scan_id-1], s = 0.1)

        # Create a scalar mappable for colorbar
        norm = mpl.colors.Normalize(vmin=run_length, vmax=run_length + (n_lines - 1) * run_length)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for colorbar

        # Create and place the colorbar
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Time / min', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.tick_params(axis='both', labelsize=8)
        ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1], xscale='log')
        
    else:
        n_lines = len(df.keys())
        cmap = mpl.colormaps['plasma_r']
        colors = cmap(np.linspace(0, 1, n_lines))
        
        data = np.array(df[df.keys()]).T
        
        if bin_edges is not None:
            dlogDp = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
            data = data / dlogDp
        
        for i in range(n_lines):
            ax.plot(bins, data[i], color=colors[i], lw=1.2)

        # Create a scalar mappable for colorbar
        norm = mpl.colors.Normalize(vmin=run_length, vmax=run_length + (n_lines - 1) * run_length)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for colorbar

        # Add colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Time / min', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.tick_params(axis='both', labelsize=8)
        ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1], xscale='log')
    
    return ax

def instrument_comparison(ax, x_data, y_data, label, ax_labels, forced_zero):
    x_plot = np.linspace(min(x_data), max(x_data), 1000)
    ax.plot(x_plot, x_plot, color = 'grey', lw = 1, ls = '--', label = None)

    if forced_zero:
        fit_params, fit_errors, squares, ndof, R2 = linear_fit(x_data, y_data, linear_forced_zero, a_guess = 1)
        y_fit = linear_forced_zero(x_plot, *fit_params)
    else:
        fit_params, fit_errors, squares, ndof, R2 = linear_fit(x_data, y_data, linear, a_guess = 1, b_guess = 0)
        y_fit = linear(x_plot, *fit_params)
    
    print(f'f(x) = {fit_params[0]}x + {fit_params[1]}, R2 = {R2}')

    ax.plot(x_plot, y_fit, color = 'k', lw = 1.2, label = 'Fit')
    ax.scatter(x_data, y_data, s=10, c='b', label = label)
    ax.legend()
    
    ax.set(xlabel = ax_labels[0], ylabel = ax_labels[1])
    
    return fit_params, fit_errors, squares, ndof, R2