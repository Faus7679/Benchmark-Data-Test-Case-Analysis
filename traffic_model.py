"""
Stochastic Traffic Data Model — Assignment 5: Benchmark-Data-Test-Case-Analysis
Simulates highway traffic using motion-sensor data modeled with appropriate
probability distributions (Poisson for vehicle counts, Gaussian for speeds,
Log-normal for headways) and visualizes the resulting time series.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── Simulation parameters ─────────────────────────────────────────────────────
HOURS = 24                     # one full day
MINUTES_PER_HOUR = 60
TOTAL_MINUTES = HOURS * MINUTES_PER_HOUR   # 1 440 one-minute intervals

start_time = datetime(2024, 1, 15, 0, 0, 0)
timestamps = [start_time + timedelta(minutes=i) for i in range(TOTAL_MINUTES)]

# ── Diurnal (time-of-day) traffic intensity curve ─────────────────────────────
# Models rush-hour peaks at ~8 AM and ~5 PM using two Gaussians summed on a
# low base-rate.  The result drives the Poisson rate λ(t).
t = np.arange(TOTAL_MINUTES) / MINUTES_PER_HOUR   # fractional hours 0–24
morning_peak  = 18 * np.exp(-0.5 * ((t -  8.0) / 1.0) ** 2)
evening_peak  = 15 * np.exp(-0.5 * ((t - 17.0) / 1.2) ** 2)
base_rate     = 3.0
lambda_t      = base_rate + morning_peak + evening_peak   # vehicles / minute

# ── 1. Vehicle counts — Poisson distribution ──────────────────────────────────
# The number of vehicles detected by a sensor in a fixed interval is classically
# Poisson-distributed (memoryless arrivals, rare events in short windows).
vehicle_counts = rng.poisson(lam=lambda_t)

# ── 2. Vehicle speeds — Gaussian (Normal) distribution ───────────────────────
# Under free-flow conditions individual vehicle speeds are approximately normally
# distributed.  During congested periods (high count) mean speed drops and
# variance narrows.  We model each minute's mean speed as inversely related to
# vehicle density.
base_speed_mean  = 65.0   # mph, free-flow
base_speed_std   = 8.0    # mph
congestion_factor = np.clip(vehicle_counts / lambda_t.max(), 0.0, 1.0)
speed_mean = base_speed_mean * (1.0 - 0.45 * congestion_factor)
speed_std  = base_speed_std  * (1.0 - 0.30 * congestion_factor) + 2.0
avg_speeds = rng.normal(loc=speed_mean, scale=speed_std)
avg_speeds = np.clip(avg_speeds, 5.0, 90.0)   # physical bounds

# ── 3. Vehicle headways — Log-normal distribution ────────────────────────────
# Time gaps (headways) between successive vehicles are right-skewed and bounded
# below by a minimum reaction distance; log-normal is the standard choice.
# When counts are high, headways shrink (lower μ_ln).
mu_ln    = np.log(np.maximum(60.0 / np.maximum(vehicle_counts, 0.5), 1.0))
sigma_ln = 0.50
headways = rng.lognormal(mean=mu_ln, sigma=sigma_ln)
headways = np.clip(headways, 0.5, 120.0)   # seconds

# ── 4. Sensor noise — Gaussian white noise ───────────────────────────────────
# Motion sensors introduce zero-mean Gaussian measurement error.
sensor_noise = rng.normal(loc=0.0, scale=0.5, size=TOTAL_MINUTES)
noisy_speeds = avg_speeds + sensor_noise

# ── Visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.suptitle(
    "Simulated Highway Traffic — 24-Hour Time Series\n"
    "(Stochastic Model: Poisson Counts · Gaussian Speeds · Log-Normal Headways)",
    fontsize=13, fontweight="bold", y=0.98
)

hour_locator  = mdates.HourLocator(interval=2)
hour_formatter = mdates.DateFormatter("%H:%M")

# Panel A — Vehicle counts (Poisson)
ax = axes[0]
ax.fill_between(timestamps, vehicle_counts, alpha=0.6, color="steelblue")
ax.plot(timestamps, lambda_t, color="navy", linewidth=1.5,
        linestyle="--", label="λ(t) — Poisson rate")
ax.set_ylabel("Vehicles / min", fontsize=10)
ax.set_title("A.  Vehicle Counts  [Poisson(λ(t))]", fontsize=10, loc="left")
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle=":", alpha=0.5)

# Panel B — Average speeds (Gaussian)
ax = axes[1]
ax.plot(timestamps, noisy_speeds, color="darkorange", linewidth=0.8, label="Speed + sensor noise")
ax.plot(timestamps, avg_speeds,   color="red",        linewidth=1.5, label="Mean speed μ(t)")
ax.set_ylabel("Speed (mph)", fontsize=10)
ax.set_title("B.  Vehicle Speeds  [Normal(μ(t), σ(t))]", fontsize=10, loc="left")
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle=":", alpha=0.5)

# Panel C — Headways (Log-normal)
ax = axes[2]
ax.fill_between(timestamps, headways, alpha=0.6, color="seagreen")
ax.set_ylabel("Headway (s)", fontsize=10)
ax.set_title("C.  Vehicle Headways  [Log-Normal(μ_ln(t), σ_ln=0.50)]", fontsize=10, loc="left")
ax.grid(axis="y", linestyle=":", alpha=0.5)

# Panel D — Cumulative vehicles (integrated Poisson process)
ax = axes[3]
cumulative = np.cumsum(vehicle_counts)
ax.plot(timestamps, cumulative, color="purple", linewidth=1.5)
ax.set_ylabel("Cumulative Vehicles", fontsize=10)
ax.set_title("D.  Cumulative Vehicle Count  (Integrated Poisson Process)", fontsize=10, loc="left")
ax.grid(axis="y", linestyle=":", alpha=0.5)

for ax in axes:
    ax.xaxis.set_major_locator(hour_locator)
    ax.xaxis.set_major_formatter(hour_formatter)
axes[-1].set_xlabel("Time of Day (HH:MM)", fontsize=10)

plt.tight_layout()
output_path = "traffic_time_series.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved → {output_path}")
plt.close()

# ── Distribution histograms ───────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle("Marginal Distributions of Simulated Traffic Variables", fontsize=12, fontweight="bold")

axes2[0].hist(vehicle_counts, bins=30, color="steelblue", edgecolor="white", density=True)
axes2[0].set_title("Vehicle Counts\n(Poisson)", fontsize=10)
axes2[0].set_xlabel("Count")
axes2[0].set_ylabel("Density")

axes2[1].hist(noisy_speeds, bins=40, color="darkorange", edgecolor="white", density=True)
axes2[1].set_title("Vehicle Speeds\n(Gaussian)", fontsize=10)
axes2[1].set_xlabel("Speed (mph)")

axes2[2].hist(headways, bins=40, color="seagreen", edgecolor="white", density=True)
axes2[2].set_title("Vehicle Headways\n(Log-Normal)", fontsize=10)
axes2[2].set_xlabel("Headway (s)")

plt.tight_layout()
hist_path = "traffic_distributions.png"
plt.savefig(hist_path, dpi=150, bbox_inches="tight")
print(f"Distribution plot saved → {hist_path}")
plt.close()

# ── Summary statistics ────────────────────────────────────────────────────────
print("\n── Summary Statistics ──────────────────────────────────────────")
print(f"Total simulated vehicles : {vehicle_counts.sum():,}")
print(f"Peak vehicles/min        : {vehicle_counts.max()} "
      f"(at {timestamps[vehicle_counts.argmax()].strftime('%H:%M')})")
print(f"Mean speed               : {avg_speeds.mean():.1f} mph  "
      f"(std = {avg_speeds.std():.1f})")
print(f"Median headway           : {np.median(headways):.1f} s  "
      f"(IQR = {np.percentile(headways, 75) - np.percentile(headways, 25):.1f})")
