import numpy as np
import pandas as pd
#%%

def running_mean(df, dictkey, concentration, timelabel, interval, wndw, timestamps):
    # Set 'Time' as the index
    new_df = pd.DataFrame() 

    if timestamps == None:
        new_df[timelabel] = pd.to_datetime(df[timelabel])
        new_df[dictkey] = df[concentration]
        new_df = new_df.set_index(timelabel)

        # Resample the data to bins 
        new_df = new_df.resample(interval).mean() 
        
        # Now, apply the rolling mean
        new_df[dictkey] = new_df[dictkey].rolling(window = wndw, min_periods = 1).mean()

    if timestamps != None:
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

def bin_mean(timestamps, df, df_keys, timelabel, inst_error):
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
        std[i] += filtered_conc.std() / np.sqrt(len(filtered_conc))
    
    if inst_error != None:
        errors = mean * inst_error
    else:
        errors = 0

    return mean, std, errors

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

