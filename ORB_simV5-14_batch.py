# Copyright (c) 2024 Taylor A Murphree
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# See the LICENSE file or https://creativecommons.org/licenses/by-nc/4.0/ for more information.

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch plotting
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import windows
import csv
import IsoSpecPy
import os

###############################################################################
# 1) CSV Export
###############################################################################
def export_to_csv(filename, mz_values, intensities, centroid):
    """
    Writes out (m/z, Intensity) pairs. If centroid is not None, appends a 'Centroid' row.
    """
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["m/z", "Intensity"])
        for mz_val, intensity_val in zip(mz_values, intensities):
            csvwriter.writerow([mz_val, intensity_val])
        if centroid is not None:
            csvwriter.writerow(["Centroid", centroid])

###############################################################################
# 2) Compute Isotopic Distribution (Simple Example)
###############################################################################
def compute_isotopic_distribution(formula, d2o_percentage):
    """
    Simplified function to generate a small set of isotopic peaks (M, M+13C, M+2H, etc.).
    Adjust or expand for your real molecule if needed.
    """
    iso = IsoSpecPy.Iso(formula)
    mono_mass = iso.getMonoisotopicPeakMass()

    d_fraction = d2o_percentage / 100.0
    h_fraction = 1.0 - d_fraction

    # Basic pattern with partial overlap between M+13C and M+2H
    isotopic_pattern = [
        (mono_mass, 100 * h_fraction),
        (mono_mass + 1.00335, 21 * 1.1 * h_fraction),
        (mono_mass + 1.00628, 100 * d_fraction),
        (mono_mass + 1.00335 + 1.00628, 21 * 1.1 * d_fraction)
    ]
    return isotopic_pattern

###############################################################################
# 3) Ensemble-Based Time-Domain Simulation
###############################################################################
def simulate_transient_ensemble(isotopic_pattern,
                                k_true, scale_factor,
                                transient_length, sampling_rate,
                                n_ensemble=200,
                                freq_dev=1e-5,
                                partial_coherence=0.2):
    """
    Summation of cosine waves for an ensemble of ions to approximate Orbitrap signals.
    """
    t = np.linspace(0, transient_length,
                    int(transient_length * sampling_rate),
                    endpoint=False)
    transient = np.zeros_like(t)

    k_sim = k_true / scale_factor

    for (mz, peak_intensity) in isotopic_pattern:
        freq_nominal = k_sim / mz
        ion_intensity = peak_intensity / n_ensemble

        for _ in range(n_ensemble):
            # fractional shift in frequency
            delta_f = np.random.normal(loc=0.0, scale=freq_dev)
            freq_ion = freq_nominal * (1.0 + delta_f)

            # partial_coherence=0 => all ions in same phase, >0 => random offset in [-X, +X]
            phase_range = partial_coherence * np.pi
            phase_shift = np.random.uniform(-phase_range, phase_range)

            transient += ion_intensity * np.cos(2.0 * np.pi * freq_ion * t + phase_shift)

    return t, transient

###############################################################################
# 4) Apply Hann Window
###############################################################################
def apply_hann_window(transient):
    hann_window = windows.hann(len(transient))
    return transient * hann_window

###############################################################################
# 5) Compute Spectrum (FFT -> m/z)
###############################################################################
def compute_spectrum(transient, sampling_rate, k_true, scale_factor,
                     mz_center, mz_window):
    fft_result = fft(transient)
    freqs = fftfreq(len(transient), d=1.0 / sampling_rate)

    # Keep positive frequencies only
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    magnitudes = np.abs(fft_result[pos_mask])

    k_scaled = k_true / scale_factor
    mz_array = k_scaled / freqs

    # Filter around [mz_center - mz_window, mz_center + mz_window]
    mz_min = mz_center - mz_window
    mz_max = mz_center + mz_window
    valid_mask = (mz_array >= mz_min) & (mz_array <= mz_max)

    mz_filtered = mz_array[valid_mask]
    intensity_filtered = magnitudes[valid_mask]

    if mz_filtered.size == 0:
        return mz_filtered, intensity_filtered, None

    # Compute centroid
    centroid = np.sum(mz_filtered * intensity_filtered) / np.sum(intensity_filtered)
    return mz_filtered, intensity_filtered, centroid

###############################################################################
# 6) Main Simulation Function
###############################################################################
def run_orbitrap_simulation(formula, resolution,
                            d2o_list,
                            k_true=6.79e13,
                            scale_factor=1e6,
                            sampling_rate=6e6,
                            n_ensemble=200,
                            freq_dev=1e-5,
                            partial_coherence=0.2,
                            mz_window=4.0,
                            output_folder="batch_sim_outputs",
                            do_plot=False):
    """
    Automates the entire pipeline for one (formula, resolution) across multiple D₂O percentages.
    - Writes CSV for each D₂O fraction.
    - Optionally generates a single combined plot (one PNG) per (formula, resolution).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # scale transient length by resolution for example
    transient_length = resolution / 120000.0

    iso = IsoSpecPy.Iso(formula)
    mono_mass = iso.getMonoisotopicPeakMass()
    mz_center = mono_mass

    plot_data = []

    for d2o in d2o_list:
        isotopic_pattern = compute_isotopic_distribution(formula, d2o)

        t, transient = simulate_transient_ensemble(
            isotopic_pattern,
            k_true=k_true,
            scale_factor=scale_factor,
            transient_length=transient_length,
            sampling_rate=sampling_rate,
            n_ensemble=n_ensemble,
            freq_dev=freq_dev,
            partial_coherence=partial_coherence
        )

        transient_windowed = apply_hann_window(transient)

        mz, intensity, centroid = compute_spectrum(
            transient_windowed,
            sampling_rate,
            k_true,
            scale_factor,
            mz_center,
            mz_window
        )

        if mz.size == 0:
            # skip if no data
            continue

        # unique CSV name
        out_csv = f"{formula}_res{resolution}_D2O_{d2o:.4f}.csv"
        out_csv_path = os.path.join(output_folder, out_csv)
        export_to_csv(out_csv_path, mz, intensity, centroid)

        # store for optional plotting
        plot_data.append((d2o, mz, intensity, centroid))

    # optional single PNG per combination
    if do_plot and len(plot_data) > 0:
        plt.figure(figsize=(10, 6))
        for (d2o, mz, intensity, centroid) in plot_data:
            lbl = f"{d2o:.4f}% D₂O, Centroid={centroid:.3f}"
            plt.plot(mz, intensity, label=lbl)
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.title(f"Orbitrap Simulation: {formula} @ Res={resolution}")
        plt.legend()
        png_name = f"{formula}_res{resolution}_summary.png"
        plt.savefig(os.path.join(output_folder, png_name))
        plt.close()

###############################################################################
# 7) The Pairs of (resolution, formula)
###############################################################################
# Based on the list you provided:
pairs = [
    (5404,   "C21H17N4O4"),
    (21642,  "C21H17N4O4"),
    (79826,  "C21H17N4O4"),
    (165022, "C21H17N4O4"),

    (5825,   "C21H18N3O2"),
    (22826,  "C21H18N3O2"),
    (85779,  "C21H18N3O2"),
    (205053, "C21H18N3O2"),

    (5758,   "C22H18F3N2"),
    (21921,  "C22H18F3N2"),
    (81326,  "C22H18F3N2"),
    (171331, "C22H18F3N2"),

    (6881,   "C17H17N2"),
    (26790,  "C17H17N2"),
    (102333, "C17H17N2"),
    (282690, "C17H17N2"),

    (7947,   "C13H13N2"),
    (28072,  "C13H13N2"),
    (110250, "C13H13N2"),
    (287049, "C13H13N2"),
]

###############################################################################
# 8) Run All Simulations in a Batch
###############################################################################
if __name__ == "__main__":

    # Decide what D₂O levels you want
    d2o_list = [0.0156, 10.013, 20.0105, 35.0066, 50.0028, 86.993]  # or any set

    # Loop over each pair
    for (res_val, formula) in pairs:
        print(f"\n=== Running simulation for formula={formula}, resolution={res_val} ===")
        run_orbitrap_simulation(
            formula=formula,
            resolution=res_val,
            d2o_list=d2o_list,
            output_folder="batch_sim_outputs",
            do_plot=False  # set True if you want a single PNG per pair
        )

    print("\nAll batch simulations complete!")
