import numpy as np
import csv
import math
from scipy.optimize import curve_fit
from scipy.special import erf

###############################################################################
# 1) Define the Two Models of Interest
###############################################################################

def gaussian(x, a, x0, sigma):
    """Symmetrical Gaussian."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def lorentzian_asymmetric(x, a, x0, sigma, lambda_):
    """
    A simple Lorentzian-like function with an asymmetry factor.
    """
    return a / (
        1 + ((x - x0) / (sigma * (1 + lambda_ * (x - x0))))**2
    )

###############################################################################
# 2) The 20 Training Datasets
#    Each tuple: (resolution_label_string, numpy_array_of_6_xy_points)
###############################################################################

data_sets = [
    ("5758",    np.array([[0.000156, 0],       [0.1001304, 0.022833333],
                          [0.2001048, 0.004],  [0.350664, 0.008],
                          [0.500028, 0.019],   [0.869933, 0.007]])),
    ("21921",   np.array([[0.000156, 0],       [0.1001304, 0.007],
                          [0.2001048, 0.034],  [0.350664, 0.025],
                          [0.500028, 0.029],   [0.869933, 0.002]])),
    ("81326",   np.array([[0.000156, 0],       [0.1001304, 0.021],
                          [0.2001048, 0.054],  [0.350664, 0.029],
                          [0.500028, 0.052],   [0.869933, 0.002]])),
    ("171331",  np.array([[0.000156, 0],       [0.1001304, 0.018],
                          [0.2001048, 0.022],  [0.350664, 0.018],
                          [0.500028, 0.019],   [0.869933, 0.008]])),

    ("5404",    np.array([[0.000156, 0],       [0.1001304, 0.027],
                          [0.2001048, 0.016],  [0.350664, 0.03],
                          [0.500028, 0.063],   [0.869933, 0.0256]])),
    ("21642",   np.array([[0.000156, 0],       [0.1001304, 0.028],
                          [0.2001048, 0.012],  [0.350664, 0.059],
                          [0.500028, 0.0054],  [0.869933, 0.0428]])),
    ("79826",   np.array([[0.000156, 0],       [0.1001304, 0.014],
                          [0.2001048, 0.014],  [0.350664, 0.048],
                          [0.500028, 0.036],   [0.869933, 0.0032]])),
    ("165022",  np.array([[0.000156, 0],       [0.1001304, 0.006],
                          [0.2001048, 0.005],  [0.350664, 0.044],
                          [0.500028, 0.02],    [0.869933, 0.006]])),

    ("6112",    np.array([[0.000156, 0],       [0.1001304, 0.004],
                          [0.2001048, 0.023],  [0.350664, 0.03],
                          [0.500028, 0.031],   [0.869933, 0.007]])),
    ("24387",   np.array([[0.000156, 0],       [0.1001304, 0.007],
                          [0.2001048, 0.025],  [0.350664, 0.031],
                          [0.500028, 0.03],    [0.869933, 0.009]])),
    ("92789",   np.array([[0.000156, 0],       [0.1001304, 0.035],
                          [0.2001048, 0.051],  [0.350664, 0.05],
                          [0.500028, 0.04],    [0.869933, 0.004]])),
    ("220448",  np.array([[0.000156, 0],       [0.1001304, 0.022],
                          [0.2001048, 0.034],  [0.350664, 0.042],
                          [0.500028, 0.036],   [0.869933, 0.002]])),

    ("6881",    np.array([[0.000156, 0],       [0.1001304, 0.006],
                          [0.2001048, 0.01],   [0.350664, 0.013],
                          [0.500028, 0.019],   [0.869933, 0.009]])),
    ("26790",   np.array([[0.000156, 0],       [0.1001304, 0.022],
                          [0.2001048, 0.019],  [0.350664, 0.005],
                          [0.500028, 0.01],    [0.869933, 0.018]])),
    ("102333",  np.array([[0.000156, 0],       [0.1001304, 0.009],
                          [0.2001048, 0.016],  [0.350664, 0.033],
                          [0.500028, 0.034],   [0.869933, 0.013]])),
    ("282690",  np.array([[0.000156, 0],       [0.1001304, 0.023],
                          [0.2001048, 0.039],  [0.350664, 0.0376],
                          [0.500028, 0.022],   [0.869933, 0.0461]])),

    ("5825",    np.array([[0.000156, 0],       [0.1001304, 0.008],
                          [0.2001048, 0.014],  [0.350664, 0.029],
                          [0.500028, 0.041],   [0.869933, 0.003]])),
    ("22826",   np.array([[0.000156, 0],       [0.1001304, 0.016],
                          [0.2001048, 0.017],  [0.350664, 0.025],
                          [0.500028, 0.027],   [0.869933, 0.006]])),
    ("85779",   np.array([[0.000156, 0],       [0.1001304, 0.018],
                          [0.2001048, 0.027],  [0.350664, 0.016],
                          [0.500028, 0.023],   [0.869933, 0.007]])),
    ("205053",  np.array([[0.000156, 0],       [0.1001304, 0.018],
                          [0.2001048, 0.023],  [0.350664, 0.035],
                          [0.500028, 0.009],   [0.869933, 0.014]]))
]

###############################################################################
# 3) Fit Each Dataset to Both Models, Store the Parameters
###############################################################################

def fit_gaussian(x, y):
    # initial guess: amplitude, center, sigma
    p0 = [max(y), np.mean(x), np.std(x)]
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=p0)
        return popt
    except:
        return None

def fit_lorentzian_asymmetric(x, y):
    # initial guess: amplitude, center, sigma, lambda
    p0 = [max(y), np.mean(x), np.std(x), 0.1]
    try:
        popt, _ = curve_fit(lorentzian_asymmetric, x, y, p0=p0)
        return popt
    except:
        return None

# We'll store them in param_tables, keyed by model name:
param_table_gauss = []   # Will be list of (resolution, [params...])
param_table_loras = []   # Will be list of (resolution, [params...])

def resolution_as_float(lbl):
    """Convert dataset label (e.g., '5758') to float. Adjust if needed."""
    return float(lbl)

for (lbl, xy) in data_sets:
    x = xy[:, 0]
    y = xy[:, 1]
    res_val = resolution_as_float(lbl)

    gauss_params = fit_gaussian(x, y)
    loras_params = fit_lorentzian_asymmetric(x, y)

    # Only store if fit succeeded:
    if gauss_params is not None:
        param_table_gauss.append( (res_val, gauss_params) )
    if loras_params is not None:
        param_table_loras.append( (res_val, loras_params) )

# Sort by resolution to make interpolation simpler
param_table_gauss.sort(key=lambda x: x[0])
param_table_loras.sort(key=lambda x: x[0])

###############################################################################
# 4) Function to Interpolate Parameters (Linear) for a Given Resolution
###############################################################################

def interpolate_params(param_table, target_resolution):
    """
    param_table is a list of (res_float, [p1, p2, p3, ...]).
    We'll linearly interpolate each parameter dimension vs. res_float.
    Returns a list [p1_interp, p2_interp, ...].
    If target_resolution is outside the range, np.interp will clamp to boundary.
    """
    if len(param_table) < 2:
        return None  # not enough data to interpolate meaningfully

    # separate the table into arrays
    res_arr = np.array([pt[0] for pt in param_table], dtype=float)
    # Assume all param sets have the same dimension as the first one
    param_dim = len(param_table[0][1])
    interp_params = []

    for j in range(param_dim):
        pj_arr = np.array([pt[1][j] for pt in param_table], dtype=float)
        # linear interpolation for param j
        pj_target = np.interp(target_resolution, res_arr, pj_arr)
        interp_params.append(pj_target)

    return interp_params

###############################################################################
# 5) Prompt for a New Resolution, Interpolate, Then Generate Y-Values
###############################################################################

def main():
    # Ask for new resolution from user
    user_input = input("Enter an Orbitrap resolution (e.g. 150000): ").strip()
    try:
        test_resolution = float(user_input)
    except ValueError:
        print("Invalid input, must be a number.")
        return

    # Interpolate for both models
    gauss_p = interpolate_params(param_table_gauss, test_resolution)
    loras_p = interpolate_params(param_table_loras, test_resolution)

    if gauss_p is None:
        print("Not enough data to interpolate Gaussian parameters.")
        return
    if loras_p is None:
        print("Not enough data to interpolate Lorentzian-Asymmetric parameters.")
        return

    # Create the big set of X values
    # For clarity, let's put them in a list. (You provided a big range from 0.000156 to 1.000156, stepping by 0.001)
    # We'll replicate your entire sequence here:
    x_values = []
    current = 0.000156
    while current <= 1.000156 + 1e-9:  # a small tolerance
        x_values.append(current)
        current += 0.001

    # Generate Y-values for each model
    # gauss_p is [a, x0, sigma]
    # loras_p is [a, x0, sigma, lambda_]
    y_gauss_list = [gaussian(x, *gauss_p) for x in x_values]
    y_loras_list = [lorentzian_asymmetric(x, *loras_p) for x in x_values]

    # Write to CSV
    # We'll place a header row with resolution & parameters, then data rows
    output_filename = "interpolation_output.csv"
    with open(output_filename, mode="w", newline="") as f:
        writer = csv.writer(f)

        # First line: input resolution + parameter sets
        # Something like: ["Resolution", test_resolution, "GaussParams", G1, G2, G3, "LorAsymParams", L1, L2, L3, L4]
        header_row = [
            "UserResolution", test_resolution,
            "GaussianParams"
        ]
        header_row += list(gauss_p)
        header_row += [
            "LorentzAsymParams"
        ]
        header_row += list(loras_p)
        writer.writerow(header_row)

        # Second line: We'll write column headers for X, Y_Gaussian, Y_LorAsym
        writer.writerow(["X", "Y_Gaussian", "Y_LorentzAsym"])

        # Then one line per X
        for xv, yg, yl in zip(x_values, y_gauss_list, y_loras_list):
            writer.writerow([xv, yg, yl])

    print(f"Done. Wrote interpolated parameters and {len(x_values)} predicted Y-values to {output_filename}.")


if __name__ == "__main__":
    main()
