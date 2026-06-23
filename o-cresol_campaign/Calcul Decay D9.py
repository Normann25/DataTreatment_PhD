import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.stats import t

# =========================
# File
# =========================
lv_path = Path(
    r"O:\Nat_Chem-Aerosol-data\Data\Processed Data\Campaign"
    r"\Cresol Campaign\20260623_Test_H2O2_o-cresol_light\PTRMS"
    r"\20260623_d9-butanol_traces.txt"
)

# =========================
# Parameters to change manually
# =========================
fit_start = "2026-06-23 13:25:00"   # Light ON
fit_end   = "2026-06-23 15:20:00"   # Light OFF

average_time = "1min"   # examples: "30s", "1min", "2min", "5min"

confidence = 0.95

k_D9B = 2.7e-12         # cm3 molecule-1 s-1

# =========================
# Load data
# =========================
df = pd.read_csv(lv_path, sep="\t")

df["DateTime"] = pd.to_datetime(
    df["AbsTime"],
    unit="D",
    origin="1899-12-30"
)

signal_col = "m67.129 (C4D9OH_i) (Raw)"

df = df[["DateTime", signal_col]].dropna()
df = df[df[signal_col] > 0]

# =========================
# Average data
# =========================
df = (
    df.set_index("DateTime")
    .resample(average_time)[signal_col]
    .mean()
    .dropna()
    .reset_index()
)

# =========================
# Select fitting period
# =========================
sub = df.loc[
    (df["DateTime"] >= pd.to_datetime(fit_start)) &
    (df["DateTime"] <= pd.to_datetime(fit_end))
].copy()

if len(sub) < 3:
    raise ValueError("Not enough data points in the selected fit period.")

# =========================
# Calculate ln(I/I0)
# =========================
sub["time_s"] = (
    sub["DateTime"] - sub["DateTime"].iloc[0]
).dt.total_seconds()

I0 = sub[signal_col].iloc[0]

sub["ln_I_I0"] = np.log(
    sub[signal_col] / I0
)

# =========================
# Linear fit
# =========================
x = sub["time_s"].values
y = sub["ln_I_I0"].values

slope, intercept = np.polyfit(x, y, 1)

y_fit = intercept + slope * x
sub["fit"] = y_fit

# =========================
# Regression statistics
# =========================
n = len(x)
dof = n - 2

residuals = y - y_fit

s_err = np.sqrt(
    np.sum(residuals**2) / dof
)

x_mean = np.mean(x)

Sxx = np.sum(
    (x - x_mean)**2
)

slope_se = s_err / np.sqrt(Sxx)

alpha = 1 - confidence

tcrit = t.ppf(
    1 - alpha / 2,
    dof
)

# =========================
# Confidence interval on slope
# =========================
slope_low = slope - tcrit * slope_se
slope_high = slope + tcrit * slope_se

# =========================
# OH concentration and uncertainty
# =========================
OH = -slope / k_D9B

OH_low = -slope_high / k_D9B
OH_high = -slope_low / k_D9B

OH_uncertainty = (OH_high - OH_low) / 2

# =========================
# Confidence interval on fitted line
# =========================
y_ci = tcrit * s_err * np.sqrt(
    1 / n +
    ((x - x_mean) ** 2 / Sxx)
)

sub["ci_lower"] = y_fit - y_ci
sub["ci_upper"] = y_fit + y_ci

# =========================
# R²
# =========================
ss_res = np.sum((y - y_fit) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)

R2 = 1 - ss_res / ss_tot

# =========================
# Print results
# =========================
print(f"Averaging time = {average_time}")
print(f"Fit start = {fit_start}")
print(f"Fit end = {fit_end}")
print(f"Number of points = {n}")
print("")
print(f"Slope = {slope:.3e} s-1")
print(
    f"Slope {confidence*100:.0f}% CI = "
    f"[{slope_low:.3e}, {slope_high:.3e}] s-1"
)
print("")
print(
    f"OH = ({OH:.2e} ± {OH_uncertainty:.2e}) "
    f"molecule cm-3"
)
print(
    f"OH {confidence*100:.0f}% CI = "
    f"[{OH_low:.2e}, {OH_high:.2e}] molecule cm-3"
)
print(f"R² = {R2:.4f}")

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(
    sub["DateTime"],
    sub["ln_I_I0"],
    s=30,
    label=f"{average_time} average"
)

ax.plot(
    sub["DateTime"],
    sub["fit"],
    linewidth=2,
    label=(
        "Fit\n"
        f"OH = ({OH:.2e} ± {OH_uncertainty:.1e}) "
        f"molecule cm$^{{-3}}$"
    )
)

ax.fill_between(
    sub["DateTime"],
    sub["ci_lower"],
    sub["ci_upper"],
    alpha=0.25,
    label=f"{confidence*100:.0f}% CI on fit"
)

ax.set_xlabel("Time")
ax.set_ylabel("ln(I / I0)")

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%H:%M")
)

ax.xaxis.set_major_locator(
    mdates.HourLocator(interval=1)
)

plt.xticks(rotation=45)

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()

plt.savefig(
    lv_path.parent / f"OH_d9butanol_decay_{average_time}_CI.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()