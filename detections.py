import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.signal import find_peaks, peak_widths
import logging
import scipy
import re

def make_density_mask(n: np.ndarray, config: dict) -> np.ndarray:
    # Make a mask for the density of detections in the x-y plane or in xyz
    n_max = config['threshold_max_cutoff']
    n_min = config['threshold_min_cutoff']
    mask = np.zeros(n.shape)
    mask[ (n >= n_min) & (n <= n_max) ] = 1
    return mask


def make_image_index(det_xyz, x_bins, y_bins, z_bins) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_idx = np.digitize(det_xyz[:,0], x_bins)-1
    y_idx = np.digitize(det_xyz[:,1], y_bins)-1
    z_idx = np.digitize(det_xyz[:,2], z_bins)-1
    return x_idx, y_idx, z_idx

def im_to_detection_entry(im: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    # Mask detections in the x-y plane
    return im[x_idx,y_idx]

def mask_detections_2d(mask_xy: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    # Mask detections in the x-y plane
    idx = mask_xy[x_idx,y_idx]
    return idx.astype(bool)

def mask_detections_3d(mask_xyz: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray, z_idx: np.ndarray) -> np.ndarray:
    # Mask detections in the x-y plane
    idx = mask_xyz[x_idx,y_idx,z_idx]
    return idx.astype(bool)

def bin_detections(det_xyz: np.ndarray,resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logging.info("bin_detections")
    # bin detections into a 3D grid
    x = det_xyz[:,0]
    y = det_xyz[:,1]
    z = det_xyz[:,2]
    # Add an extra bin to the end for histogramdd() then remove it later
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + resolution, resolution)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y) + resolution, resolution)
    z_bins = np.arange(np.nanmin(z), np.nanmax(z) + resolution, resolution)
    counts_xyz = np.histogramdd([x, y, z], bins=[x_bins, y_bins, z_bins])
    return counts_xyz[0], x_bins[:-1], y_bins[:-1], z_bins[:-1]

def bins3d_to_stats2d(counts_xyz: np.ndarray, z_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info("bins3d_to_stats2d")
    # Calculate moments and sd from 3D bins
    n_xy = np.sum(counts_xyz,axis=2)
    moment_1_xy = np.sum(counts_xyz * z_bins, axis=2) / (n_xy + 1)
    moment_2_xy = np.sum(counts_xyz * (z_bins**2), axis=2) / (n_xy + 1)
    var_xy = moment_2_xy - moment_1_xy ** 2
    sd_xy = np.sqrt(var_xy)
    return n_xy, moment_1_xy, sd_xy

def median_by_time(df_all: pd.DataFrame, config: dict) -> np.ndarray:
    df = df_all[[config['image_id_col'], config['x_col'], config['y_col'], config['z_col']]]
    # Group by time-point and calculate the median for x, y, and z
    grouped = df.groupby(config['image_id_col']).median()
    # Extract the median values for x, y, and z
    median_values = grouped[[config['x_col'], config['y_col'], config['z_col']]].values

    ret = np.full((np.max(grouped.index)+1,3),np.nan)
    ret[grouped.index,:] = median_values
    return ret

def create_backup_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info("create_backup_columns")
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    sd_colnames = [config['x_sd_col'], config['y_sd_col'], config['z_sd_col']]
    ndims = len(xyz_colnames)
    for backup_num in range(9, 0, -1):
        for dim in range(ndims):
            col = [xyz_colnames[dim], sd_colnames[dim]]
            for c in col:
                backup_from = f"{c}_{backup_num-1}"
                backup_to = f"{c}_{backup_num}"
                if backup_from in df.columns:
                    df[backup_to] = df[backup_from]
                    logging.info(f"Copying col {backup_from} -> {backup_to}")
                if backup_num == 1:
                    backup_from = c
                    backup_to = f"{c}_0"
                    df[backup_to] = df[backup_from]
                    logging.info(f"Copying col {backup_from} -> {backup_to}")
    return df

def filter_detections(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filters rows in a DataFrame based on column names and corresponding ranges.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        config (dict): A dictionary with select_cols and select_ranges keys.
    Returns:
        pd.DataFrame: Filtered DataFrame with rows satisfying all conditions.
    """
    if 'select_cols' not in config or 'select_ranges' not in config:
        return df

    colnames = config['select_cols']
    ranges = config['select_ranges']

    if colnames == '' or ranges == '' or colnames is None or ranges is None:
        return df
    # Split colnames and ranges into lists
    columns = colnames.split(',')
    range_list = ranges.split(',')
    # Ensure columns and ranges have the same length
    if len(columns) != len(range_list):
        raise ValueError(f"The number of columns and ranges must match: {colnames} vs {ranges}")

    # Start with the full DataFrame
    filtered_df = df.copy()
    logging.info(f"Initial number of rows: {filtered_df.shape[0]}")

    # TODO: Build up idx and filter once.
    # Apply filters for each column and range
    for col, r in zip(columns, range_list):
        # Parse the range (e.g., "1.0-3.2" -> 1.0, 3.2)
        low, high = map(float, re.split(r'(?<!^)-', r.strip()))
        # Filter the DataFrame
        filtered_df = filtered_df[(filtered_df[col] >= low) & (filtered_df[col] <= high)]
        logging.info(f"Filtered {col} between {low} and {high}: {filtered_df.shape[0]} rows")
    return filtered_df

def cat_experiment(df: pd.DataFrame, df2: pd.DataFrame, df_offset: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info('cat_experiment')
    # Ensure df_offset contains only one row
    if len(df_offset) != 1:
        raise ValueError("df_offset must contain exactly one row.")
    # Assumption is that cols in df, df2 and df_offset are the same.
    # df_offset contains one row with the offsets to add.
    # This should contain the following: image_id_col, time_point_col and x_col,y_col,z_col
    logging.info(f"Offsets for adding frames: image_id_col={df_offset[config['image_id_col']].values[0]}, time_point_col={df_offset[config['time_point_col']].values[0]}")
    logging.info(f"Offsets for adding frames: x_col={df_offset[config['x_col']].values[0]}, y_col={df_offset[config['y_col']].values[0]}, z_col={df_offset[config['z_col']].values[0]}")
    cols = [config['image_id_col'], config['time_point_col'], config['x_col'], config['y_col'], config['z_col']]
    for col in cols:
        if col not in df.columns:
            logging.warning(f"Column {col} not found in df. Ignoring offset for this column.")
        else:
            df2[col] = df2[col] + df_offset[col].values[0]
    df = pd.concat([df, df2], ignore_index=True)
    return df

def apply_corrections(df: pd.DataFrame, x_t: np.ndarray, config: dict) -> pd.DataFrame:
    logging.info('apply_corrections')
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    ndimensions = len(xyz_colnames)
    for j in range(ndimensions):
        tidx = df[config['image_id_col']]
        df[xyz_colnames[j]] = df[xyz_colnames[j]] - x_t[j, tidx]
    # Correct deltaz if it exists
    if config['deltaz_col'] in df.columns:
        df[config['deltaz_col']] = df[config['deltaz_col']] - x_t[2, df[config['image_id_col']]]
    return df

def deltaz_correct_detections(df: pd.DataFrame, df_fiducials: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info('zstep_correct_detections')
    # Find x,y,z_deltaz_slope across fiducials and apply correction to all detections based on this
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    xyz_dimnames = config['dimnames']
    ndimensions = len(xyz_colnames)
    for j in range(ndimensions):
        slopes = df_fiducials[f'{xyz_dimnames[j]}_deltaz_slope'].values
        slope_median = np.median(slopes)
        # Apply the correction to all detections
        df[xyz_colnames[j]] = df[xyz_colnames[j]] - slope_median * df[config['deltaz_col']]
    # TODO: Possibly find it as a function of z also
    return df

def compute_deltaz(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info('add_deltaz_column')
    # TODO: What if z_step_col does not exist but deltaz_col does?
    # Add a column to the dataframe that is the relative z position
    if not config['deltaz_col'] in df.columns:
        logging.warning('Column deltaz does not exist, creating')
    else:
        logging.warning('Column deltaz already exists in df, overwriting')
    df[config['deltaz_col']] = df[config['z_col']] - df[config['z_step_col']] * config['z_step_step']
    return df

def compute_image_id(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info('compute_image_id')
    if not config['image_id_col'] in df.columns:
        # Check max for image-ID
        min_cycle, max_cycle = map(int, config['cycle_range'].split('-'))
        min_frame, max_frame = map(int, config['frame_range'].split('-'))
        min_z_step, max_z_step = map(int, config['z_step_range'].split('-'))
        min_time_point, max_time_point = map(int, config['time_point_range'].split('-'))
        num_frames = max_frame - min_frame + 1
        num_cycles = max_cycle - min_cycle + 1
        num_z_steps = max_z_step - min_z_step + 1
        frames_per_cycle = num_frames * num_z_steps
        # Compute the image ID from the frame number and time point
        df[config['image_id_col']] = (df[config['frame_col']] +
                                      (df[config['z_step_col']] - min_z_step) * num_frames +
                                      (df[config['cycle_col']] - min_cycle) * frames_per_cycle +
                                      (df[config['time_point_col']] - min_time_point) * frames_per_cycle * num_cycles)
    return df

def compute_time_derivates(df: pd.DataFrame, df_drift: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info('compute_time_derivatives')
    # Compute the time derivatives of x, y, z and adds them to df_drift
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    x_col = ['x', 'y', 'z']
    xyz_median = median_by_time(df, config)
    for j,col in enumerate(xyz_colnames):
        # Compute the time derivative of each column
        df_drift[f'd{x_col[j]}_dt'] = 0
        # Compute the time derivative of the median values
        deriv = np.diff(xyz_median[:,j])
        idx_deriv = np.arange(1,xyz_median.shape[0])
        # Add the derivative to df_drift in those rows where the image-ID matches
        df_drift.loc[idx_deriv, f'd{x_col[j]}_dt'] = deriv
        # Set any values spanning changes in z_step to 0
        z_step_vs_t = np.full(len(df_drift),np.nan)
        z_step_vs_t[df[config['image_id_col']].values] = df[config['z_step_col']].values
        z_step_change = np.diff(z_step_vs_t)
        z_step_change = np.insert(z_step_change, 0, 1)
        df_drift.loc[z_step_change!=0, f'd{x_col[j]}_dt'] = 0
    return df_drift

def fwhm_from_points(x):
    """
    Computes the Full Width at Half Maximum (FWHM) of a distribution
    represented by an array of x-values.

    Parameters:
        x (array-like): Array of x-values (data points).

    Returns:
        float: The FWHM value.
    """
    # Create histogram
    if len(x) == 0 or np.all(np.isnan(x)):
        return np.nan

    x = x[~np.isnan(x)]
    if len(x) <10000:
        bins =100
    else:
        bins = 1000

    counts, bin_edges = np.histogram(x, bins=bins)
    # smooth counts
    counts = scipy.ndimage.gaussian_filter1d(counts, 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks in the histogram
    peaks, _ = find_peaks(counts)

    # Find width of the highest peak at half height
    if len(peaks) > 0:
        max_peak_index = peaks[np.argmax(counts[peaks])]
        results_half = peak_widths(counts, [max_peak_index], rel_height=0.5)

        # Width in bin units — convert to x-units using bin width
        width_bins = results_half[0][0]
        bin_width = bin_edges[1] - bin_edges[0]
        width_x_units = width_bins * bin_width
        # Positions of the half-height crossings
        left_idx = int(results_half[2][0])
        right_idx = int(results_half[3][0])
        left_x = float(bin_centers[left_idx])
        right_x = float(bin_centers[right_idx])
        half_height = counts[max_peak_index] / 2
    else:
        return np.nan
    debug = 0
    if debug:
        plt.plot(bin_centers, counts)
        plt.title("Histogram with Peak")
        plt.xlabel("x")
        plt.ylabel("Counts")
        plt.axvline(bin_centers[max_peak_index], color='r', linestyle='--', label='Peak')
        plt.axhline(half_height, color='g', linestyle='--', label='Half Height')
        plt.axvline(left_x, color='b', linestyle='--', label='Left Crossing')
        plt.axvline(right_x, color='b', linestyle='--', label='Right Crossing')
        plt.legend()
        plt.show()
        plt.close()
    return width_x_units

def fwhm_from_points2(x):
    """
    Computes the Full Width at Half Maximum (FWHM) of a distribution
    represented by an array of x-values. Original jfm version.

    Parameters:
        x (array-like): Array of x-values (data points).

    Returns:
        float: The FWHM value.
    """
    # If x is zero length or all nan, return nan
    if len(x) == 0 or np.all(np.isnan(x)):
        return np.nan

    x = x[~np.isnan(x)]
    if len(x) <10000:
        bins =100
    else:
        bins = 1000
    counts, bin_edges = np.histogram(x, bins=bins, density=True)
    # smooth counts
    counts = scipy.ndimage.gaussian_filter1d(counts, 1)
    peak_index = np.argmax(counts)
    peak_value = counts[peak_index]
    half_max = peak_value / 2

    left_crossing = None
    right_crossing = None

    for i in range(len(counts) - 1):
        # Check for crossings
        if counts[i] <= half_max < counts[i + 1]:
            # Linear interpolation for left crossing
            left_crossing = bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) * (
                        (half_max - counts[i]) / (counts[i + 1] - counts[i]))
            break

    # loop backwards over counts
    for i in range(len(counts) - 1, 0, -1):
        if counts[i] >= half_max > counts[i + 1]:
            # Linear interpolation for right crossing
            right_crossing = bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) * (
                        (half_max - counts[i]) / (counts[i + 1] - counts[i]))
            break

    # Ensure crossings were found
    if left_crossing is None and right_crossing is None:
        raise ValueError("Could not find two crossings at half-maximum.")
    elif left_crossing is None:
        fwhm_value = right_crossing - np.nanmin(x)
    elif right_crossing is None:
        fwhm_value = np.nanmax(x) - left_crossing
    else:
        fwhm_value = right_crossing - left_crossing
    return fwhm_value

def check_z_step(df: pd.DataFrame, config: dict) -> int:
    logging.info('check_z_step')
    # Check that df[config['z_step_col']] actually changes and can be regressed against
    if df[config['z_step_col']].nunique() < 2:
        logging.warning(f"Column {config['z_step_col']} does not change in the data, cannot check z_step_step")
        return 1
    # Check that deltaz is within the expected range
    z_step_slope, intercept, cor, p_value, std_err = scipy.stats.linregress(df[config['z_step_col']], df[config['z_col']])
    # if the sign of z_step_step is different to the sign of z_step_slope then warn
    if np.sign(z_step_slope) != np.sign(config['z_step_step']):
        logging.warning(f"Sign of empirical z_step_step {z_step_slope} is different to sign of z_step_step {config['z_step_step']}")
        return 1
    else:
        logging.info(f"Empirical z_step_step: {z_step_slope} Config z_step_step: {config['z_step_step']}")
        return 0
