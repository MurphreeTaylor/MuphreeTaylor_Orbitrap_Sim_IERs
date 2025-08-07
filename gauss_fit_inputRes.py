import numpy as np
from scipy.interpolate import interp1d

def interpolate_parameters(resolution, resolutions, amplitudes, centers, std_devs):
    """
    Interpolates the amplitude, center, and standard deviation for a given Orbitrap resolution.

    Parameters:
        resolution (float): The desired Orbitrap resolution for interpolation.
        resolutions (list or numpy array): List of Orbitrap resolutions.
        amplitudes (list or numpy array): Corresponding amplitudes.
        centers (list or numpy array): Corresponding centers.
        std_devs (list or numpy array): Corresponding standard deviations.

    Returns:
        tuple: (interpolated amplitude, center, standard deviation)
    """
    # Create interpolation functions
    amplitude_interp = interp1d(resolutions, amplitudes, kind='linear', fill_value='extrapolate')
    center_interp = interp1d(resolutions, centers, kind='linear', fill_value='extrapolate')
    std_dev_interp = interp1d(resolutions, std_devs, kind='linear', fill_value='extrapolate')

    # Interpolate for the given resolution
    interpolated_amplitude = amplitude_interp(resolution)
    interpolated_center = center_interp(resolution)
    interpolated_std_dev = std_dev_interp(resolution)

    return interpolated_amplitude, interpolated_center, interpolated_std_dev

if __name__ == "__main__":
    # Tabulated data
    resolutions = np.array([480000, 120000, 30000, 7500])
    amplitudes = np.array([0.042998327, 0.052933959, 0.032691635, 0.033091233])
    centers = np.array([0.358464466, 0.309264207, 0.442907828, 0.445896067])
    std_devs = np.array([0.223585614, 0.245389029, 0.246678799, 0.222185537])

    # Prompt user for Orbitrap resolution
    resolution = float(input("Enter the desired Orbitrap resolution: "))

    # Interpolate parameters
    amplitude, center, std_dev = interpolate_parameters(resolution, resolutions, amplitudes, centers, std_devs)

    # Output the interpolated parameters
    print(f"Interpolated Parameters for Resolution {resolution}:")
    print(f"Amplitude: {amplitude}")
    print(f"Center: {center}")
    print(f"Standard Deviation: {std_dev}")