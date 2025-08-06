#################################################
# Ditte Thomsen, August 2025
# Import function of SASS data and data treatment to create particle size distribution. 
#################################################

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# import biblo

def import_SASS(filepath):
    # Imports SASS data: Size and spectral density from .SASS files. 
    # Parameters: filepath (str): Path to the .SASS file. 
    # Returns: 

    # Read the file 
    df = pd.read_csv(filepath, delimiter='\t', header=None, skiprows=8, engine='python')

    # List to store scans 
    scans = []
    scan_number = None
    in_data_block = False
    current_data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Start of a new scan
            if line.startswith("SCAN"):
                parts = line.split('\t')
                try:
                    scan_number = int(parts[1])
                except (IndexError, ValueError):
                    scan_number = None
                in_data_block = False  # wait for next header to begin collecting
                continue

            # Header line (start of scan data)
            if line.startswith("Scan Time"):
                in_data_block = True
                continue

            # End of a scan block
            if line.startswith("END OF SCAN"):
                if current_data:
                    # Convert collected lines to DataFrame
                    df = pd.DataFrame(current_data, columns=[
                        'ScanTime', 'RealTime', 'Size', 'SpectralDensity', 'CorrectedSpectralDensity'
                    ])
                    df['ScanNumber'] = scan_number
                    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
                    df['CorrectedSpectralDensity'] = pd.to_numeric(df['CorrectedSpectralDensity'], errors='coerce')
                    df = df[['Size', 'CorrectedSpectralDensity', 'ScanNumber']].dropna()
                    scans.append(df)
                    current_data = []
                in_data_block = False
                continue

            # Collect data lines
            if in_data_block:
                parts = line.split('\t')
                if len(parts) >= 5:
                    current_data.append(parts[:5])

    # Combine all scans into one DataFrame
    if scans:
        return pd.concat(scans, ignore_index=True)
    else:
        return pd.DataFrame(columns=['Size', 'CorrectedSpectralDensity', 'ScanNumber'])

#figfix function from "biblo" is called to streamline all figures
# biblo.figfix(AutoLayout=True)

filepath = Path(r"O:\Nat_Chem-Aerosol-data\People\Ditte Thomsen\Python\SASS\Data\AAC_Data001.SASS")

df = import_SASS(filepath)

plt.figure(figsize=(10,6))
for scan_id, group in df.groupby("ScanNumber"):
    plt.plot(group["Size"], group["CorrectedSpectralDensity"],marker='.',linestyle='None',label=f"Scan {scan_id}")

plt.xscale('log')

plt.xlabel('Size (nm)')
plt.ylabel('dN/dlogD$_a$ (cc)')
plt.legend()
plt.grid(True, which='both',ls='-',linewidth=0.5)

plt.tight_layout()
plt.show()