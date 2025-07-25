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

param_table_gauss = []   # list of (resolution_value, [gauss_params...])
param_table_loras = []   # list of (resolution_value, [loras_params...])

def resolution_as_float(lbl):
    """Convert dataset label (e.g., '5758') to float. Adjust if needed."""
    return float(lbl)

# Fit each training dataset and store results
for (lbl, xy) in data_sets:
    x = xy[:, 0]
    y = xy[:, 1]
    res_val = resolution_as_float(lbl)

    gauss_params = fit_gaussian(x, y)
    loras_params = fit_lorentzian_asymmetric(x, y)

    if gauss_params is not None:
        param_table_gauss.append( (res_val, gauss_params) )
    if loras_params is not None:
        param_table_loras.append( (res_val, loras_params) )

# Sort by resolution
param_table_gauss.sort(key=lambda x: x[0])
param_table_loras.sort(key=lambda x: x[0])

###############################################################################
# 4) Function to Interpolate Parameters (Linear) for a Given Resolution
###############################################################################

def interpolate_params(param_table, target_resolution):
    """
    param_table: list of (res_float, [p1, p2, p3, ...]).
    We'll linearly interpolate each parameter dimension vs. res_float.
    If target_resolution is outside the range, np.interp will clamp to boundary.
    Returns: list [p1_interp, p2_interp, ...] or None if not enough data.
    """
    if len(param_table) < 2:
        return None  # not enough data

    res_arr = np.array([pt[0] for pt in param_table], dtype=float)
    param_dim = len(param_table[0][1])  # how many parameters
    interp_params = []

    for j in range(param_dim):
        pj_arr = np.array([pt[1][j] for pt in param_table], dtype=float)
        pj_target = np.interp(target_resolution, res_arr, pj_arr)
        interp_params.append(pj_target)

    return interp_params

###############################################################################
# 5) Main Function: Prompt for 20 resolutions, process each, write CSV outputs
###############################################################################

def main():
    print("Please enter 20 resolution values, one per line.")
    for i in range(1, 21):
        user_input = input(f"Resolution {i} of 20: ").strip()
        try:
            test_resolution = float(user_input)
        except ValueError:
            print("Invalid input, must be a number. Skipping this entry.")
            continue

        # Interpolate for both models
        gauss_p = interpolate_params(param_table_gauss, test_resolution)
        loras_p = interpolate_params(param_table_loras, test_resolution)

        if gauss_p is None:
            print("Not enough data to interpolate Gaussian parameters.")
            continue
        if loras_p is None:
            print("Not enough data to interpolate Lorentzian-Asymmetric parameters.")
            continue

        # Generate X values from 0.000156 to 1.000156 (step=0.001)
        x_values = []
        current = 0.000156
        while current <= 1.000156 + 1e-9:
            x_values.append(current)
            current += 0.001

        # Compute Y for each model
        y_gauss_list = [gaussian(x, *gauss_p) for x in x_values]
        y_loras_list = [lorentzian_asymmetric(x, *loras_p) for x in x_values]

        # Prepare an output filename that includes the resolution
        out_filename = f"interpolation_output_{int(test_resolution)}.csv"

        with open(out_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Header row with resolution + parameters
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

            # Second line: column headers for data
            writer.writerow(["X", "Y_Gaussian", "Y_LorentzAsym"])

            # Then each row of computed values
            for xv, yg, yl in zip(x_values, y_gauss_list, y_loras_list):
                writer.writerow([xv, yg, yl])

        print(f"Done. Wrote interpolated results for resolution {test_resolution} to {out_filename}.")

if __name__ == "__main__":
    main()
