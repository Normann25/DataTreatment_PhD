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

def plot_barchart(ax, means, stds, xticks, ax_label):
    n_lines = len(xticks)
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, n_lines))

    ax.bar(xticks, means, yerr = stds, color = colors)
    ax.set_ylabel(ax_label)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha = 'left')
    ax.tick_params(axis = 'x', which = 'minor', bottom = False)

    return

def plot_multi_barchart(axes, means, stds, xticks, ax_label):
    for ax, key in zip(axes, means.keys()):
        plot_barchart(ax, means[key], stds[key], xticks, ax_label)

    return

def plot_heatmap(ax, df, df_keys, time, bin_edges, cutpoint, normed, t_zero):

    data = np.array(df[df_keys])

    if normed == False:
        dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
        data=data/dlogDp
    
    if t_zero is not None:
        time = (pd.to_datetime(time) - pd.to_datetime(t_zero)) / pd.Timedelta(minutes = 1)
        
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

    if t_zero is not None:
        ax.set_xlabel('Time (min)')
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
        ax.set_xlabel("Time (HH:MM)")
        plt.subplots_adjust(hspace=0.05)
        
    # Make the y-scal logarithmic and set a label
    ax.set_yscale("log")
    ax.set_ylabel("Dp (nm)")
    return ax, p1

def plot_total(ax, df, conc_key, clr, t_zero):
    if t_zero is not None:
        time = (df['Time'] - pd.to_datetime(t_zero)) / pd.Timedelta(minutes = 1)
        ax.plot(time, df[conc_key], lw = 1, color = clr)

        ax.set_xlabel('Time (min)')

    else:
        ax.plot(df['Time'], df[conc_key], lw = 1, color = clr)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
        ax.set_xlabel("Time (HH:MM)")
        plt.subplots_adjust(hspace=0.05)
    return ax

def plot_timeseries(fig, ax, df, df_keys, bin_edges, datatype, timestamps, normed, total, cutpoint, t_zero):

    if datatype == 'number and mass':
        df_number, df_mass = df[0], df[1]

        new_df_number = time_filtered_conc(df_number, df_keys, timestamps)
        new_df_mass = time_filtered_conc(df_mass, df_keys, timestamps)

        if total is not None:
            ax1, p1 = plot_heatmap(ax[0][0], new_df_number, df_keys, np.array(new_df_number['Time']), bin_edges, cutpoint, normed, t_zero)
            ax2, p2 = plot_heatmap(ax[0][1], new_df_mass, df_keys, np.array(new_df_mass['Time']), bin_edges, cutpoint, normed, t_zero)

            total_df_number = time_filtered_conc(df_number, [total], timestamps)
            ax3 = plot_total(ax[1], total_df_number, total, 'r', t_zero)

            total_df_mass = time_filtered_conc(df_mass, [total], timestamps)    
            ax4 = plot_total(ax[1], total_df_mass, total, 'r', t_zero)

            ax3.set_ylabel('Total number conc. (# cm$^{-3}$)')
            ax4.set_ylabel('Total mass conc. ($\mu$g m$^{-3}$)')

        else:
            ax1, p1 = plot_heatmap(ax[0], new_df_number, df_keys, np.array(new_df_number['Time']), bin_edges, cutpoint, normed, t_zero)
            ax2, p2 = plot_heatmap(ax[1], new_df_mass, df_keys, np.array(new_df_mass['Time']), bin_edges, cutpoint, normed, t_zero)

        # Insert coloarbar and label it
        col1 = fig.colorbar(p1, ax=ax1)
        col2 = fig.colorbar(p2, ax=ax2)

        col1.set_label('dN/dlogDp (# cm$^{-3}$)')
        col2.set_label('dM/dlogDp ($\mu$g m$^{-3}$)')

    else:
        new_df = time_filtered_conc(df, df_keys, timestamps)

        if total is not None:
            ax1, p1 = plot_heatmap(ax[0], new_df, df_keys, np.array(new_df['Time']), bin_edges, cutpoint, normed, t_zero)

            total_df = time_filtered_conc(df, [total], timestamps)             
            ax2 = plot_total(ax[1], total_df, total, 'r', t_zero)

        else:
            ax1, p1 = plot_heatmap(ax, new_df, df_keys, np.array(new_df['Time']), bin_edges, cutpoint, normed, t_zero)

        # Insert coloarbar and label it
        col = fig.colorbar(p1, ax=ax1)
        if datatype == "number":
            col.set_label('dN/dlogDp (# cm$^{-3}$)')
            if total != None:
                ax2.set_ylabel('Total concentration (# cm$^{-3}$)')
        elif datatype == "mass":
            col.set_label('dM/dlogDp ($\mu$g m$^{-3}$)')
            if total != None:
                ax2.set_ylabel('Total concentration ($\mu$g m$^{-3}$)')


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
    ax.set_ylabel('dN/dlogDp (# cm$^{-3}$)', color='tab:blue')

    ax.set(xlabel='Particle diameter (nm)', xscale='log')

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
        ax2.set_ylabel('dM/dlogDp ($\mu$g m$^{-3}$)', color='red')  # Use axis_labels[2] for clarity
    
    else:
        ax2, mean_mass = 0, 0
    
    return mean_number, mean_mass, ax, ax2

def plot_running_sizedist(fig, ax, df, bins, bin_edges, axis_labels, run_length):
    
    data = np.array(df[df.keys()])
    n_lines = len(data)
    cmap = mpl.colormaps['plasma_r']
    colors = cmap(np.linspace(0, 1, n_lines))
    
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
    cbar.set_label('Time (min)', fontsize=9)
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

def vanKrevelen_OS(ax, rotation):
    O_C_ratio = np.linspace(0, 1.5, 100)

    for OS in np.linspace(-2, 2, 9): # Oxidation states from -2 to 2
        H_C_ratio = 2*O_C_ratio-OS
        ax.plot(O_C_ratio, H_C_ratio, color = 'lightgray', lw = 0.75, ls = '--', zorder = -10)
    
    OS_labels = ['OS = -2', 'OS = -1', 'OS = 0', 'OS = 1']
    OS_placement = [[0.003, 2.13], [0.003, 1.12], [0.465, 1.05], [0.955, 1.03]]
    for label, placement in zip(OS_labels, OS_placement):
        ax.text(placement[0], placement[1], label, rotation = rotation, fontsize = 7, color = 'darkgray')

    # Organic compound lines
    x = np.linspace(0, 1.5, 100)
    y1 = linear(x, 0, 2) # Alcohol/peroxide
    y2 = linear(x, -1, 2) # Carboxylic acid
    y3 = linear(x, -2, 2) # Carbonyls (aldehyde/ketone)

    compound_labels = ['slope = 0 \n + alcohol/peroxide', 'slope = -1 \n + carboxylic acid', 'slope = -2 \n + carbonyls']
    compound_placements = [[0.9, 1.96], [0.66, 1.28], [0.2, 1.31]]

    for i, y in enumerate([y1, y2, y3]):
        ax.plot(x, y, color = 'darkgray', lw = 0.75, zorder = -10)
        ax.text(compound_placements[i][0], compound_placements[i][1], 
                compound_labels[i], 
                bbox=dict(ec = 'gray', fc = 'white', lw = 0.5, pad = 0.5),
                color = 'gray', fontsize = 6)

    ax.set(xlim = (0,1.5), ylim = (1,2.55), xlabel = 'O:C', ylabel = 'H:C')
    return ax

def vanKrevelen_ts(df, df_keys, timestamps, run_length):
    conc_mask = df[df_keys[2]] >= 0.03 # Based on AMS detection limit for organics (in V-mode)
    df = df[conc_mask]

    new_df = time_filtered_conc(df, df_keys, timestamps)

    n_points = len(new_df['Time'])
    cmap = mpl.colormaps['viridis_r']
    fig, ax = plt.subplots(1,2, figsize = (6.3, 3))

    c_ = np.linspace(1, n_points, n_points)
    ax[0].scatter(new_df[df_keys[1]], new_df[df_keys[0]], c = c_, cmap = cmap, s = 10)
    ax[1].scatter(new_df[df_keys[1]], new_df[df_keys[0]], c = c_, cmap = cmap, s = 10)

    # Create a scalar mappable for colorbar
    norm = mpl.colors.Normalize(vmin=run_length, vmax=run_length + (n_points - 1) * run_length)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for colorbar

    # Add colorbar to the figure
    cbar = fig.colorbar(sm, ax=ax[1], orientation='vertical')
    cbar.set_label('Time (min)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    vanKrevelen_OS(ax[0], 63.5)

    ax[1].set(xlabel = 'O:C', ylabel = 'H:C')

    return fig, ax

def vanKrevelen_multi_exp(ax, data_dict, dict_keys, df_keys, timestamps, labels):
    n_exp = len(dict_keys)
    cmap = mpl.colormaps['viridis_r']
    colors = cmap(np.linspace(0, 1, n_exp+1))
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for i, key in enumerate(dict_keys):
        conc_mask = data_dict[key][df_keys[2]] >= 0.03 # Based on AMS detection limit for organics (in V-mode)
        df = data_dict[key][conc_mask]

        new_df = time_filtered_conc(df, df_keys, timestamps[i])
        rho = density_from_AMS(new_df[df_keys[0]], new_df[df_keys[1]])

        ax[0].scatter(new_df[df_keys[1]], new_df[df_keys[0]], color = colors[i], s = 10, marker = markers[i])
        ax[1].scatter(new_df[df_keys[2]], rho, color = colors[i], s = 10, marker = markers[i])

    ax[1].legend(labels = labels, fontsize = 8, frameon = True)   #, bbox_to_anchor = (1, 0, 0, 1))
    vanKrevelen_OS(ax[0], 60)
    ax[1].set(xlabel = 'Org conc ($\mu$g m$^{-3}$)', ylabel = 'Density (kg m$^{-3}$)')

    return ax

def plot_SMPS(data, dictkeys, df_keys, min_DP, datatype, timestamps, run_length, total_key, t_zero, nrows, ncols, save_path):
    bin_edges = [min_DP]
    for key in df_keys:
        bin_edges.append(float(key))
    
    running_SMPS = {}
    for i, time in enumerate(timestamps):
        if datatype == 'number and mass':
            temp_number = running_mean(data[dictkeys[0][i]], df_keys, 'Time', f'{run_length}T', run_length, [t_zero, time[1]])
            running_SMPS[dictkeys[0][i]] = temp_number
            temp_mass = running_mean(data[dictkeys[1][i]], df_keys, 'Time', f'{run_length}T', run_length, [t_zero, time[1]])
            running_SMPS[dictkeys[1][i]] = temp_mass
        else:
            temp = running_mean(data[dictkeys[i]], df_keys, 'Time', f'{run_length}T', run_length, [t_zero, time[1]])
            running_SMPS[dictkeys[i]] = temp
        
    fig_run_number, ax_run_number = plt.subplots(nrows, ncols, figsize = (3.5*ncols, 3*nrows))
    if 'mass' in datatype:
        fig_run_mass, ax_run_mass = plt.subplots(nrows, ncols, figsize = (3.5*ncols, 3*nrows))
    fig_mean, ax_mean = plt.subplots(nrows, ncols, figsize = (3.5*ncols, 3*nrows))
    axes_number, axes_mass = [], []

    for i, time in enumerate(timestamps):
        if datatype == 'number and mass':
            fig1, ax1 = plt.subplots(2, 1, figsize = (6.3, 6))
            plot_timeseries(fig1, ax1, data[dictkeys[0][i]], df_keys, bin_edges, 'number', time, True, total_key, None, t_zero)
            fig1.tight_layout()
            fig1.savefig(f'{save_path}Timeseries_{dictkeys[0][i]}.jpg', dpi = 600)
            fig2, ax2 = plt.subplots(2, 1, figsize = (6.3, 6))
            plot_timeseries(fig2, ax2, data[dictkeys[1][i]], df_keys, bin_edges, 'mass', time, True, total_key, None, t_zero)
            fig2.tight_layout()
            fig2.savefig(f'{save_path}Timeseries_{dictkeys[1][i]}.jpg', dpi = 600)

            if nrows > 1 or ncols > 1:
                ax = ax_mean.flatten()[i]
                ax_number, ax_mass = ax_run_number.flatten()[i], ax_run_mass.flatten()[i]
            else:
                ax = ax_mean
                ax_number, ax_mass = ax_run_number, ax_run_mass
            number, mass, ax3, ax3_2 = plot_bin_mean(ax, [t_zero, time[1]], data[dictkeys[0][i]], data[dictkeys[1][i]], df_keys, 'Time', bin_edges[1:], None, None, True)
            fig_mean.tight_layout()
            fig_mean.savefig(f'{save_path}SizeDist_{dictkeys[0][i]}.jpg', dpi = 600)
            axes_number.append(ax3)
            axes_mass.append(ax3_2)

            plot_running_sizedist(fig_run_number, ax_number, running_SMPS[dictkeys[0][i]], bin_edges[1:], None, ['Diameter (nm)', 'dN/dlogDp (# cm$^{-3}$)'], run_length)
            fig_run_number.tight_layout()
            fig_run_number.savefig(f'{save_path}Running_SizeDist_{dictkeys[0][i]}.jpg', dpi = 600)
            plot_running_sizedist(fig_run_mass, ax_mass, running_SMPS[dictkeys[1][i]], bin_edges[1:], None, ['Diameter (nm)', 'dM/dlogDp ($\mu$g m$^{-3}$)'], run_length)
            fig_run_mass.tight_layout()
            fig_run_mass.savefig(f'{save_path}Running_SizeDist_{dictkeys[1][i]}.jpg', dpi = 600)

        else:
            fig1, ax1 = plt.subplots(2, 1, figsize = (6.3, 6))
            plot_timeseries(fig1, ax1, data[dictkeys[i]], df_keys, bin_edges, datatype, time, True, total_key, None, t_zero)
            fig1.tight_layout()
            fig1.savefig(f'{save_path}Timeseries_{dictkeys[i]}.jpg', dpi = 600)

            if datatype == 'number':
                if nrows > 1 or ncols > 1:
                    ax = ax_mean.flatten()[i]
                    ax_number = ax_run_number.flatten()[i]
                else:
                    ax = ax_mean
                    ax_number = ax_run_number

                number, mass, ax2, ax2_2 = plot_bin_mean(ax, [t_zero, time[1]], data[dictkeys[i]], None, df_keys, 'Time', bin_edges[1:], None, None, False)
                fig_mean.tight_layout()
                fig_mean.savefig(f'{save_path}SizeDist_{dictkeys[i]}.jpg', dpi = 600)
                axes_number.append(ax2)

                plot_running_sizedist(fig_run_number, ax_number, running_SMPS[dictkeys[i]], bin_edges[1:], None, ['Diameter (nm)', 'dN/dlogDp (# cm$^{-3}$)'], run_length)
                fig_run_number.tight_layout()
                fig_run_number.savefig(f'{save_path}Running_SizeDist_{dictkeys[i]}.jpg', dpi = 600)

    return axes_number, axes_mass

def plot_SASS(df, timestamps, run_length, datatype, name):

    def plot_SASS_heatmap(fig, axes, df, datatype):
        time = np.array(df['Time'])

        data = np.array([df['CorrectedSpectralDensity'][df['ScanNumber'] == 1].tolist()])
        for scan_id, group in df.groupby('ScanNumber'):
            if scan_id != 1:
                conc = group['CorrectedSpectralDensity'].tolist()
                if 596 < len(conc) < 599:
                    conc.append(0)                
                if (scan_id % 2) == 0:
                    conc = conc[::-1]                              
                data = np.concatenate((data, np.array([conc])), axis = 0)

        data = np.array(data[:max(df['ScanNumber'])])
        
        # Generate an extra time bin, which is needed for the meshgrid
        time = np.unique(time)
        new_time = time - pd.Timedelta(minutes = 10)
        new_time = np.append(new_time, time[-1])     

        size = df['Size'][df['ScanNumber'] == 1].tolist() + [111.9]
        
        # generate 2d meshgrid for the x, y, and z data of the 3D color plot
        y, x = np.meshgrid(np.array(size), new_time)

        # Set the upper and/or lower limit of the color scale based on input
        y_min = np.nanmin(data)
        y_max = np.nanmax(data)
        
        # Fill the generated mesh with particle concentration data
        p1 = axes[0].pcolormesh(x, y, data, cmap='jet',vmin=y_min, vmax=y_max,shading='flat')
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Dp (nm)")
        
        total_conc = calc_total_conc(df, [111, 5490], 202.5, ['Size', 'CorrectedSpectralDensity'])
        axes[1].plot(total_conc['Time'], total_conc['Total Concentration'], lw = 1, color = 'r')

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
            ax.set_xlabel("Time (HH:MM)")
            plt.subplots_adjust(hspace=0.05)

        if datatype == 'number':
            col = fig.colorbar(p1, ax=axes[0])
            col.set_label('dN/dlogDp (# cm$^{-3}$)')
            axes[1].set_ylabel('Total concentration (# cm$^{-3}$)')

        if datatype == 'mass':
            col = fig.colorbar(p1, ax=axes[0])
            col.set_label('dM/dlogDp ($\mu$g m$^{-3}$)')
            axes[1].set_ylabel('Total concentration ($\mu$g m$^{-3}$)')
        
    def SASS_running_SizeDist(fig, ax, df, run_length, axis_labels):
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
        cbar.set_label('Time (min)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.tick_params(axis='both', labelsize=8)
        ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1], xscale='log')
        return
    
    ax_labels = ['dN/dlogDp (# cm$^{-3}$)', 'dM/dlogDp ($\mu$g m$^{-3}$)']
    for i, dtype in enumerate(datatype):
        if timestamps == None:
            new_df = df[i][['Time', 'CorrectedSpectralDensity', 'Size', 'ScanNumber']]
        else:
            new_df = time_filtered_conc(df[i], ['CorrectedSpectralDensity', 'Size', 'ScanNumber'], timestamps)

        fig1, axes1 = plt.subplots(2, 1, figsize = (6.3, 6))
        plot_SASS_heatmap(fig1, axes1, new_df, dtype)
        fig1.tight_layout()
        fig1.savefig(f'Figures/SASS/Timeseries_{name}_{dtype}.jpg', dpi = 600)

        fig2, ax2 = plt.subplots(figsize = (4,3))
        SASS_running_SizeDist(fig2, ax2, new_df, run_length, ['Particle diameter (nm)', ax_labels[i]])
        fig2.tight_layout()
        fig2.savefig(f'Figures/SASS/SizeDist_{name}_{dtype}.jpg', dpi = 600)

    return