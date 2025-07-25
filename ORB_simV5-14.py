# Copyright (c) 2024 Taylor A Murphree
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# See the LICENSE file or https://creativecommons.org/licenses/by-nc/4.0/ for more information.

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import IsoSpecPy
from scipy.signal import windows
import csv

###############################################################################
# CSV export function that supports centroid
###############################################################################
def export_to_csv(filename, mz_values, intensities, centroid):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["m/z", "Intensity"])
        for mz_val, intensity_val in zip(mz_values, intensities):
            csvwriter.writerow([mz_val, intensity_val])
        if centroid is not None:
            csvwriter.writerow(["Centroid", centroid])

###############################################################################
# 1) Isotopic Distribution
###############################################################################
def compute_isotopic_distribution(formula, d2o_percentage):
    """
    Compute isotopic distribution for a given molecular formula and %D₂O.
    Returns a list of (mz, intensity) tuples.
    """
    iso = IsoSpecPy.Iso(formula)
    mono_mass = iso.getMonoisotopicPeakMass()

    d_fraction = d2o_percentage / 100.0
    h_fraction = 1.0 - d_fraction

    # Example pattern: monoisotopic + small shifts for M+1, M+2, etc.
    isotopic_pattern = [
        (mono_mass, 100 * h_fraction),
        (mono_mass + 1.00335, 21 * 1.1 * h_fraction),
        (mono_mass + 1.00628, 100 * d_fraction),
        (mono_mass + 1.00335 + 1.00628, 21 * 1.1 * d_fraction)
    ]

    print(f"Isotopic pattern for {d2o_percentage}% D₂O: {isotopic_pattern}")
    return isotopic_pattern

###############################################################################
# 2) Ensemble-Based Transient Simulation
###############################################################################
def simulate_transient_ensemble(isotopic_pattern,
                                k_true, scale_factor,
                                transient_length, sampling_rate,
                                n_ensemble=100,
                                freq_dev=1e-5,
                                partial_coherence=0.2):
    """
    Simulate the time-domain transient by treating each isotopic peak as an
    *ensemble of ions*, each with a slightly different frequency (freq_dev)
    and a partially coherent phase (partial_coherence).

    :param isotopic_pattern: List of (mz, intensity) for each isotopic peak.
    :param k_true: The canonical Orbitrap constant, e.g. 6.79e13.
    :param scale_factor: Factor to scale k so freq < Nyquist for 6 MHz.
    :param transient_length: In seconds (scaled by resolution).
    :param sampling_rate: e.g. 6e6
    :param n_ensemble: Number of ions to simulate per isotopic peak.
    :param freq_dev: Standard deviation of random fractional frequency shift,
                     e.g. 1e-5 means ± 0.001% shift typical.
    :param partial_coherence: 0 => all ions in phase,
                              1 => phases uniform in [0, 2π],
                              anything in-between => small random offset around 0.

    :return: (time_array, transient_signal)
    """
    print("Simulating transient with an ensemble of ions per isotope (partial coherence).")

    # Effective scaled k
    k_sim = k_true / scale_factor

    # Create time array
    t = np.linspace(0, transient_length,
                    int(transient_length * sampling_rate),
                    endpoint=False)
    transient = np.zeros_like(t)

    for (mz, peak_intensity) in isotopic_pattern:
        # Nominal frequency for this isotopic peak
        freq_nominal = k_sim / mz

        # Distribute total peak intensity among the ensemble
        ion_intensity = peak_intensity / n_ensemble

        for _ in range(n_ensemble):
            # Small random fractional shift in frequency
            delta_f = np.random.normal(loc=0.0, scale=freq_dev)  # fractional
            freq_ion = freq_nominal * (1.0 + delta_f)

            # Partial coherence => phases in [ -partial_coherence*pi, +partial_coherence*pi ]
            phase_range = partial_coherence * np.pi
            phase_shift = np.random.uniform(-phase_range, phase_range)

            # Add each ion's contribution to the transient
            transient += ion_intensity * np.cos(2.0 * np.pi * freq_ion * t + phase_shift)

    print(f"Transient signal simulated. Max amplitude: {np.max(transient):.6f}")
    return t, transient

###############################################################################
# 3) Hann (Hanning) Window
###############################################################################
def apply_hann_window(transient):
    """
    Apply a Hann window to the time-domain signal to reduce spectral leakage.
    """
    hann_window = windows.hann(len(transient))
    return transient * hann_window

###############################################################################
# 4) Compute Spectrum
###############################################################################
def compute_spectrum(transient, sampling_rate, k_true, scale_factor,
                     mz_center, mz_window):
    """
    FFT -> magnitude spectrum -> frequency -> m/z using same scale_factor => filter.
    Returns (mz_filtered, intensity_filtered, centroid).
    """
    fft_result = fft(transient)
    freqs = fftfreq(len(transient), d=1.0 / sampling_rate)

    # Keep only positive frequencies
    positive_indices = freqs > 0
    freqs = freqs[positive_indices]
    intensities = np.abs(fft_result[positive_indices])

    # Convert freq_sim -> m/z = (k_true/scale_factor) / freq_sim
    k_scaled = k_true / scale_factor
    mz = k_scaled / freqs

    # Filter around mz_center ± mz_window
    mz_min, mz_max = mz_center - mz_window, mz_center + mz_window
    valid_indices = (mz >= mz_min) & (mz <= mz_max)

    mz_filtered = mz[valid_indices]
    intensity_filtered = intensities[valid_indices]

    if mz_filtered.size == 0:
        return mz_filtered, intensity_filtered, None

    # Compute centroid
    centroid = np.sum(mz_filtered * intensity_filtered) / np.sum(intensity_filtered)
    return mz_filtered, intensity_filtered, centroid

###############################################################################
# Main Script
###############################################################################
if __name__ == "__main__":
    formula = input("Enter the molecular formula (e.g., C21H19N2): ")
    resolution = float(input("Enter the desired resolution (e.g., 120000): "))

    # Transient length in seconds, scaled by resolution
    transient_length = resolution / 120000.0
    print(f"Using transient_length = {transient_length:.6f} s")

    sampling_rate = 6e6  # 6 MHz sampling
    d2o_percentages = [0.0156, 10.013, 20.0105, 35.0066, 50.0028, 86.993]
    mz_window = 4.0

    # Canonical Orbitrap constant
    k_true = 6.79e13
    # Scale factor for dimensionless approach
    scale_factor = 1e6

    # Larger ensemble, partial coherence, random freq shifts
    n_ensemble = 200      # Number of ions per isotopic peak
    freq_dev = 1e-5       # Fractional frequency standard deviation
    partial_coherence = 0.2  # Phase range fraction of π

    # Determine monoisotopic mass => approximate center
    iso = IsoSpecPy.Iso(formula)
    mono_mass = iso.getMonoisotopicPeakMass()
    mz_center = mono_mass

    for d2o in d2o_percentages:
        print(f"\nSimulating for {d2o}% D₂O...")
        isotopic_pattern = compute_isotopic_distribution(formula, d2o)

        # 1) Simulate the time-domain signal with an ensemble approach
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

        # 2) Apply Hann window
        transient_windowed = apply_hann_window(transient)

        # 3) FFT => spectrum => filter => centroid
        mz, intensity, centroid = compute_spectrum(
            transient_windowed,
            sampling_rate,
            k_true=k_true,
            scale_factor=scale_factor,
            mz_center=mz_center,
            mz_window=mz_window
        )

        # Save and plot if data found
        if mz.size > 0:
            filename = f"{formula}_{d2o}_D2O_ensemble_spectrum.csv"
            export_to_csv(filename, mz, intensity, centroid)
            plt.plot(mz, intensity, label=f"{d2o}% D₂O, Centroid: {centroid:.4f}")
        else:
            print(f"No data in range for {d2o}% D₂O.")

    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title("Orbitrap Simulation (Ensemble + Small Freq Shifts + Partial Coherence)")
    plt.legend()
    plt.show()
