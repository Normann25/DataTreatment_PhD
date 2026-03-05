
# Avaa_api code by Pasi Kolari?
# Plotting by Kira Ryhti-Laine 27.1.2026
    # I have used python version 3.11.11 for plotting, some of the packages did not work with older versions (such as 3.9.7)
    # I have also used Visual Studio Code (VS Code) for running the code

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Install (in Anaconda Prompt etc.) required packages if not already installed:

# pip install avaa_api
# pip install pandas
# pip install datetime
# pip install requests
# pip install matplotlib
# pip install numpy
# pip install mpl_fold_axis

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import datetime, avaa_api # Avaa API package
import numpy as np # NumPy package
import pandas as pd # Pandas package
import matplotlib.pyplot as plt # Matplotlib package
import matplotlib.ticker as ticker # Matplotlib ticker package
import matplotlib.dates as mdates # Matplotlib dates package
from mpl_fold_axis import fold_axis # Fold axis package

#~~~~~~~~~~~~

yrs=[2022,2023,2024] # Years
fdate=[1,1] # First day
ldate=[12,31] # Last day
tablevars=['HYY_META.T168','HYY_META.tsoil_5','HYY_META.wsoil_5','HYY_EDDY233.NEE','HYY_EDDY233.GPP'] # Variables: Air temperature, soil temperature (5 cm), soil water content (5 cm), NEE, GPP
interval=60 # Minutes
agg='ARITHMETIC' # Aggregate with "arithmetic mean"
# leave quality undefined = use default value
#~~~~~~~~~~~~
tablevars2=['HYY_META.Precipacc'] # Precipitation variable
agg2='SUM' # Aggregate with "sum"
#~~~~~~~~~~~~
# for yy in yrs:
#    print('%d' % yy)   
#    d1=datetime.datetime(yy,fdate[0],fdate[1])    
#    d2=datetime.datetime(yy,ldate[0],ldate[1])
#    data=avaa_api.getData(d1,d2,tablevars,interval=interval,aggregation=agg)

d1=datetime.datetime(yrs[0],fdate[0],fdate[1]) # start date   
d2=datetime.datetime(yrs[2],ldate[0],ldate[1]) # end date
data =avaa_api.getData(d1,d2,tablevars,interval=interval,aggregation=agg) # get temperature, soil water content and flux data
data2=avaa_api.getData(d1,d2,tablevars2,interval=interval,aggregation=agg2) # get precipitation data

data=data.join(data2['HYY_META.Precipacc']) # join two dataframes

print(data.head()) # print first rows of the data to check

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.rcParams['font.size'] = '15' # Set global font size to change size of the y axis values

fig, ax1 = plt.subplots(4, 1, figsize=(10, 15), sharex=True) # create 4 subplots in one column, share x-axis
fig.subplots_adjust(hspace=0.05) # adjust space between subplots

# First plot - soil and air temperature

ax1[0].plot(data['Datetime'], data['HYY_META.T168'], label='Air', color='black') # plot air temperature
ax1[0].plot(data['Datetime'], data['HYY_META.tsoil_5'], label='Soil', color='orange') # plot soil temperature
# axs[0].set_title('Air and Soil Temperature at Hyytiälä (2022-2024)') # title
ax1[0].set_ylabel('Temperature (°C)', labelpad=10) # ylabel with some space
ax1[0].axhline(y=0, color='lightgrey', linestyle='-', linewidth=0.8) # horizontal line , label='Freezing Point' 
#ax1[0].set_xlim(datetime.datetime(2021, 11, 30), datetime.datetime(2025, 1, 31)) # x-axis limits
ax1[0].tick_params(axis='x', which='minor', tick1On=False, tick2On=False) # remove minor ticks
ax1[0].legend(frameon=False, loc='upper right', fontsize=12) # legend without frame
ax1[0].annotate("A", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=20) # subplot label

# Second plot with broken y-axis - soil water content

ax1[1].plot(data['Datetime'], data['HYY_META.wsoil_5'], label='SWC', color='turquoise') # plot soil water content
ax1[1].set_ylabel('Soil water content ($\mathregular{m^{3} m^{-3}}$)', labelpad=15) # ylabel with some space
#axs[1].set_title('Soil water content at Hyytiälä (2022-2024)') # title
#ax1[1].set_xlim(datetime.datetime(2021, 11, 30), datetime.datetime(2025, 1, 31)) # x-axis limits
ax1[1].tick_params(axis='x', which='minor', tick1On=False, tick2On=False) # remove minor ticks
ax1[1].legend(frameon=False, loc='upper right', fontsize=12) # legend without frame
fold_axis(ax1[1], [(0.03, 0.07, 0.25)], axis="y", which="lower") # create broken y-axis
ax1[1].set_yticks([0.1, 0.2, 0.3, 0.4]) # set y-axis ticks
ax1[1].set_ylim(0, 0.44) # y-axis limits
ax1[1].annotate("B", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=20) # subplot label

# Third plot - Net ecosystem CO2 exchange (NEE) and gross primary production(GPP)

ax1[2].plot(data['Datetime'], data['HYY_EDDY233.NEE'], label='NEE', color='black') # plot NEE
ax1[2].plot(data['Datetime'], data['HYY_EDDY233.GPP'], label='GPP', color='olivedrab') # plot GPP
#axs[2].set_title('NEE and GPP at Hyytiälä (2022-2024)') # title
ax1[2].set_ylabel('Flux (µmol $\mathregular{m^{-2} s^{-1}}$)', labelpad=6) # ylabel with some space
ax1[2].axhline(y=0, color='lightgrey', linestyle='-', linewidth=0.8) # horizontal line, label='Freezing Point' 
#ax1[2].set_xlim(datetime.datetime(2021, 11, 30), datetime.datetime(2025, 1, 31)) # x-axis limits
ax1[2].tick_params(axis='x', which='minor', tick1On=False, tick2On=False) # remove minor ticks
ax1[2].legend(frameon=False, loc='upper right', fontsize=12) # legend without frame
ax1[2].annotate("C", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=20) # subplot label

# Fourth plot - Precipitation

ax1[3].plot(data['Datetime'], data['HYY_META.Precipacc'], label='Rain', color='darkcyan') # plot precipitation
# axs[3].set_title('Precipitation at Hyytiälä (2022-2024)') # title
ax1[3].set_ylabel('Precipitation (mm)', labelpad=17) # ylabel with some space
ax1[3].axhline(y=0, color='lightgrey', linestyle='-', linewidth=0.8) # horizontal line, label='Freezing Point' 
ax1[3].set_xlim(datetime.datetime(2021, 11, 30), datetime.datetime(2025, 1, 31)) # x-axis limits
ax1[3].legend(frameon=False, loc='upper right', fontsize=12) # legend without frame
ax1[3].set_yticks([0, 5, 10, 15, 20]) # set y-axis ticks
ax1[3].xaxis.set_major_locator(mdates.YearLocator()) # set x-axis major ticks
ax1[3].xaxis.set_major_formatter(ticker.NullFormatter()) # remove major tick labels
ax1[3].xaxis.set_minor_locator(mdates.MonthLocator(7)) # set x-axis minor ticks to July of each year
ax1[3].xaxis.set_minor_formatter(mdates.DateFormatter('%Y')) # set x-axis minor tick labels to year format
ax1[3].tick_params(axis='x', which='minor', tick1On=False, tick2On=False) # remove minor ticks
ax1[3].annotate("D", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=20) # subplot label

ax1[3].set_xlabel('Years', labelpad=15) # xlabel with some space

fig.tight_layout()
fig.savefig('Courses/2026_INAR_winter_school/Figures/preassignment.jpg', dpi = 600)
plt.show() # show plots

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


