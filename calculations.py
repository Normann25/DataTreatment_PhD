import numpy as np
import pandas as pd
import scipy
from iminuit import Minuit
from ExternalFunctions import *
#%%
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

def running_mean(df, concentration, timelabel, interval, wndw, timestamps):

    if timestamps == None:      
        df = df.set_index(timelabel)

        # Resample the data to bins 
        new_df = df[concentration].resample(interval).mean() 
        
        for key in concentration:
            # Now, apply the rolling mean
            new_df[key] = new_df[key].rolling(window = wndw, min_periods = 1).mean()

    else:
        new_df = pd.DataFrame()
        
        start_time = pd.to_datetime(timestamps[0])
        end_time = pd.to_datetime(timestamps[1])

        time = pd.to_datetime(df[timelabel])

        time_filter = (time >= start_time) & (time <= end_time)

        filtered_time = pd.to_datetime(np.array(time[time_filter]))

        new_df = pd.DataFrame({'Timestamp': filtered_time})
        
        # Dictionary to collect new columns
        new_columns = {}

        for key in concentration:
            conc = np.array(df[key])
            conc = pd.to_numeric(conc, errors='coerce')
            filtered_conc = conc[time_filter]

            # Store the filtered data in the dictionary
            new_columns[key] = filtered_conc

        # Convert dictionary to DataFrame and concatenate it with `new_df`
        new_df = pd.concat([new_df, pd.DataFrame(new_columns)], axis=1)
        new_df = new_df.set_index('Timestamp')

        # Resample the data to bins 
        new_df = new_df.resample(interval).mean() 
        
        for key in concentration:
            # Now, apply the rolling mean
            new_df[key] = new_df[key].rolling(window = wndw, min_periods = 1).mean()

    return new_df

def bin_mean(timestamps, df, df_keys, timelabel):
    mean = np.zeros(len(df_keys))
    std = np.zeros(len(df_keys))

    start_time = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[1])
    if timelabel != None:
        time = pd.to_datetime(df[timelabel])
    else:
        time = pd.to_datetime(df.index)
    time_filter = (time >= start_time) & (time <= end_time)

    for i, key in enumerate(df_keys):
        conc = np.array(df[key])
        
        # Convert the concentration data to numeric, coercing errors
        conc = pd.to_numeric(conc, errors='coerce')
        filtered_conc = conc[time_filter]
        mean[i] += filtered_conc.mean()
        std[i] += filtered_conc.std() #/ np.sqrt(len(filtered_conc))

    return mean, std

def calc_mass_conc(df, df_keys, bin_mid_points, rho):
    try:
        new_df = pd.DataFrame({'Time': df['Time']})
    except KeyError:
        new_df = pd.DataFrame({'Time': df.index})

    new_columns = {}    
    for i, key in enumerate(df_keys):
        # Ensure df[key] is numeric
        df[key] = np.array(pd.to_numeric(df[key], errors='coerce'))
        
        new_columns[key] = (rho / 10**6) * (np.pi / 6) * bin_mid_points[i]**3 * df[key] * 10**6 # in ug * m**-3
    # Convert dictionary to DataFrame and concatenate it with `new_df`
    new_df = pd.concat([new_df, pd.DataFrame(new_columns)], axis=1)

    return new_df

def bin_edges(d_min, bin_mid):
    bins_list = [d_min]

    for i, bin in enumerate(bin_mid):
        bin_max = bin**2 / bins_list[i] 
        bins_list.append(bin_max)
    
    return bins_list

def calc_total_conc(df, size_bins, norm, SASS_specific):
    new_df = pd.DataFrame({'Time': df['Time'].unique()})
    
    Total_conc = np.zeros(len(new_df['Time']))
    
    if SASS_specific is not None:
        for scan_id, group in df.groupby('ScanNumber'):
            mask = (group[SASS_specific[0]] > size_bins[0]) & (group[SASS_specific[0]] < size_bins[1])
            temp = group[mask]
            #Total_conc[scan_id-1] += sum(pd.to_numeric(temp[SASS_specific[1]])) / norm
            Total_conc[scan_id-1] += abs(scipy.integrate.simpson(np.array(temp[SASS_specific[1]]) / norm, x = temp[SASS_specific[0]]))

    else:
        temp = df[size_bins]
        Dp = []
        for label in size_bins:
            Dp.append(float(label))
        
        for i, row in temp.iterrows():
            if norm is not None:
                #Total_conc[i] += sum(row) / norm
                Total_conc[i] += scipy.integrate.simpson(np.array(row) / norm, x = Dp)
            else: 
                Total_conc[i] += sum(row)
        
    new_df['Total Concentration'] = Total_conc
    
    return new_df

def density_from_AMS(H_C_ratio, O_C_ratio):
    rho = 1000 * (12 + 1*H_C_ratio + 16*O_C_ratio) / (7.0 + 5.0*H_C_ratio + 4.15*O_C_ratio) # Density in kg/m^3
    return rho

def linear_forced_zero(x, a):
    return (a * x)

def linear(x, a, b):
    return b + (a * x)

def linear_fit(x, y, fitfunc, **kwargs):

    Npoints = len(y)
    x, y = np.array(x), np.array(y)
    
    def obt(*args):
        squares = np.sum(((y-fitfunc(x, *args)))**2)
        return squares

    minuit = Minuit(obt, **kwargs, name = [*kwargs]) # Setup; obtimization function, initial variable guesses, names of variables. 
    minuit.errordef = 1 # needed for likelihood fits. No explaination in the documentation.

    minuit.migrad() # Compute the fit
    valuesfit = np.array(minuit.values, dtype = np.float64) # Convert to numpy
    errorsfit = np.array(minuit.errors, dtype = np.float64) # Convert to numpy
    # if not minuit.valid: # Give custom error if the fit did not converge
    #     print("!!! Fit did not converge !!!\n!!! Give better initial parameters !!!")

    Nvar = len(kwargs)           # Number of variables
    Ndof_fit = len(x) - Nvar

    squares_fit = minuit.fval  

    # Calculate R2
    R2 = ((Npoints * np.sum(x * y) - np.sum(x) * np.sum(y)) / (np.sqrt(Npoints * np.sum(x**2) - (np.sum(x))**2)*np.sqrt(Npoints * np.sum(y**2) - (np.sum(y))**2)))**2 

    return valuesfit, errorsfit, Ndof_fit, squares_fit, R2