#!/usr/bin/env python3
import numpy as np
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import weighted
import scipy.ndimage
import scipy.stats
import os
import logging
import multiprocessing
from zedtool.detections import im_to_detection_entry, fwhm_from_points, apply_corrections
from zedtool.plots import plot_histogram, plot_scatter, plotly_scatter
from zedtool.plots import construct_plot_path


def fit_fiducial_step_parallel(i, k, fitting_intervals, x_ft, xsd_ft, config):
    # Assumes that x_ft and xsd_ft are 1D arrays with indexing by dim and fiducial already done
    x_fit_ft = np.zeros_like(x_ft)
    x_fit_ft.fill(np.nan)
    xsd_fit_ft = np.zeros_like(xsd_ft)
    xsd_fit_ft.fill(np.nan)

    for j in range(len(fitting_intervals) - 1):
        idx = np.arange(fitting_intervals[j], fitting_intervals[j + 1])
        logging.info(f'fit_fiducial_step: fid_idx: {i} dim_idx: {k} seg_idx: {j} ')
        x_fit_ft[idx], xsd_fit_ft[idx] = fit_fiducial_step(x_ft[idx], xsd_ft[idx], config)
        y = x_ft[idx]
        ysd = xsd_ft[idx]
        y_fit = x_fit_ft[idx]
        if np.sum(~np.isnan(y)) == 0 or np.sum(~np.isnan(ysd)) == 0 or np.sum(~np.isnan(y_fit)) == 0:
            logging.warning(
                f'No valid data for fitting in fit_fiducial_detections() for fiducial {i} dimension {k} interval {j}')
            if j > 0:
                x_fit_ft[idx] = x_fit_ft[fitting_intervals[j] - 1]
                xsd_fit_ft[idx] = xsd_fit_ft[fitting_intervals[j] - 1]
            else:
                x_fit_ft[idx] = 0
                xsd_fit_ft[idx] = 0
                continue
        elif config['plot_per_fiducial_fitting']:
            plot_fiduciual_step_fit(i, j, k, x_ft[idx], xsd_ft[idx], x_fit_ft[idx],xsd_fit_ft[idx], config)

    for j in range(len(fitting_intervals) - 1, 0, -1):
        idx = np.arange(fitting_intervals[j - 1], fitting_intervals[j])
        if np.sum(xsd_fit_ft[idx] == 0) > 0:
            xsd_fit_ft[idx] = xsd_fit_ft[fitting_intervals[j]]
            x_fit_ft[idx] = x_fit_ft[fitting_intervals[j]]

    return i, k, x_fit_ft, xsd_fit_ft

def plot_fiduciual_step_fit(fiducial_index: int, interval_index: int, dimension_index: int, y: np.ndarray, ysd: np.ndarray, y_fit: np.ndarray, ysd_fit: np.ndarray, config: dict) -> int:
    logging.info('plot_fiduciual_step_fit')
    dimnames = config['dimnames']
    dim = dimnames[dimension_index]
    outdir = os.path.join(config['output_dir'], "fiducial_step_fit")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"fidx_{fiducial_index}_i_{interval_index}_d_{dim}_fit")
    x = np.arange(len(y))
    plt.figure(figsize=(10, 6))
    if np.sum(~np.isnan(y)) == 0 or np.sum(~np.isnan(ysd)) == 0 or np.sum(~np.isnan(y_fit)) == 0:
        logging.warning(f'No valid data for fitting in plot_fiduciual_step_fit() for fiducial {fiducial_index} dimension {dim} interval {interval_index}')
        return
    sc = plt.scatter(x, y, c = ysd, s = 0.1, label='Original Data')
    plt.colorbar(sc, label='sd')
    plt.scatter(x, y_fit+ysd_fit, s=0.1, label='fit+sd')
    plt.scatter(x, y_fit-ysd_fit, s=0.1, label='fit-sd')
    plt.scatter(x, y_fit, s=0.1, label='fit')
    plt.xlabel('image-ID')
    plt.ylabel(f"{dim} (nm)")
    plt.title(f'Fit for {dim}  fid={fiducial_index} tp={interval_index}')
    plt.legend()
    plt.savefig(outpath)
    plt.close()
    return 0

def fit_fiducial_step(xt: np.ndarray, xt_sd: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    polynomial_degree = config['polynomial_degree']
    use_weights_in_fit = (config['use_weights_in_fit']!=0)
    extrapolate_to_end = True
    median_filter_size = 100
    outlier_threshold = 3

    w = 1 / xt_sd
    non_nan_indices = ~np.isnan(xt) & ~np.isnan(w) & ~np.isinf(w)
    nan_indices = ~non_nan_indices
    if np.sum(non_nan_indices) == 0:
        logging.warning('No valid data for fitting in fit_fiducial_step()')
        return np.full_like(xt, np.nan), np.full_like(xt, np.nan)

    first_non_nan = np.min(np.where(non_nan_indices))
    last_non_nan = np.max(np.where(non_nan_indices))
    x = np.arange(len(xt))
    y = xt

    # Fit a polynomial using weights
    if use_weights_in_fit:
        coefficients = np.polyfit(x[non_nan_indices], y[non_nan_indices], polynomial_degree, w=w[non_nan_indices])
    else:
        coefficients = np.polyfit(x[non_nan_indices], y[non_nan_indices], polynomial_degree)
    x_fit = np.polyval(coefficients, x)

    # Calculate error bars of the fit for later use when weighting fits
    residuals = np.abs(y - x_fit)
    outlier_sd = np.nanmean(residuals[non_nan_indices]) * outlier_threshold
    residuals_filled = np.where(nan_indices, outlier_sd, residuals)
    smoothed_residuals = scipy.ndimage.median_filter(residuals_filled, size=median_filter_size)
    xsd_fit = np.copy(smoothed_residuals)
    xsd_fit[xsd_fit>outlier_threshold | np.isnan(residuals)] = outlier_sd

    if not extrapolate_to_end:
        if first_non_nan > 0:
            x_fit[:first_non_nan] = x_fit[first_non_nan]
            xsd_fit[:first_non_nan] = np.nanmean(xsd_fit[non_nan_indices]) * 3
        if last_non_nan < len(x_fit):
            x_fit[last_non_nan:] = x_fit[last_non_nan]
            xsd_fit[last_non_nan:] = np.nanmean(xsd_fit[non_nan_indices]) * 3

    return x_fit, xsd_fit

def fit_fiducial_detections_parallel(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('fit_fiducial_detections_parallel')
    ndim = x_ft.shape[0]
    nfiducials = x_ft.shape[1]
    min_cycle, max_cycle = map(int, config['cycle_range'].split('-'))
    min_frame, max_frame = map(int, config['frame_range'].split('-'))
    min_z_step, max_z_step = map(int, config['z_step_range'].split('-'))
    min_time_point, max_time_point = map(int, config['time_point_range'].split('-'))
    num_time_points = max_time_point - min_time_point + 1
    num_frames = max_frame - min_frame + 1
    num_cycles = max_cycle - min_cycle + 1
    num_z_steps = max_z_step - min_z_step + 1
    total_cycles = num_cycles * num_time_points
    frames_per_cycle = num_frames * num_z_steps
    total_frames = total_cycles * frames_per_cycle
    frames_per_time_point = num_cycles * frames_per_cycle
    fitting_intervals = np.arange(0, total_frames + frames_per_time_point, frames_per_time_point)

    x_fit_ft = np.zeros_like(x_ft)
    x_fit_ft.fill(np.nan)
    xsd_fit_ft = np.zeros_like(xsd_ft)
    xsd_fit_ft.fill(np.nan)

    with multiprocessing.Pool() as pool:
        results = pool.starmap(fit_fiducial_step_parallel,
                               [(i, k, fitting_intervals, x_ft[k,i,:], xsd_ft[k,i,:], config) for i in range(nfiducials) for k in
                                range(ndim)])

    for i, k, x_fit, xsd_fit in results:
        x_fit_ft[k, i, :] = x_fit
        xsd_fit_ft[k, i, :] = xsd_fit

    if np.sum(xsd_fit_ft == 0) > 0:
        logging.error('Zeros in xsd_fit_ft')
    if np.sum(np.isnan(x_fit_ft)) > 0:
        logging.error('NaN in x_fit_ft')
    return x_fit_ft, xsd_fit_ft

def zstep_correct_fiducials_parallel(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('zstep_correct_fiducials_parallel')
    # If x1,... are taken then move them to x2,... first.
    # Check if backup column x_0 exists, if not then quit
    if not 'x_0' in df.columns:
        logging.error('No backup columns found in df')
        return df_fiducials, df
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    ndims = len(xyz_colnames)
    nfiducials = len(df_fiducials)
    fiducial_names = df_fiducials['name']
    fiducial_labels = df_fiducials['label']

    tasks = [(fiducial_labels[j], fiducial_names[j],
              df[df['label']==fiducial_labels[j]],df['label']==fiducial_labels[j], config) for j in range(nfiducials)]

    with multiprocessing.Pool() as pool:
        results = pool.starmap(zstep_correct_fiducial, tasks)

    for df_cor, idx_cor in results:
        df.loc[idx_cor] = df_cor

    return df_fiducials, df


def zstep_correct_fiducial(fiducial_label: int, fiducial_name: str, df: pd.DataFrame, idx_cor: np.ndarray, config: dict) -> Tuple [pd.DataFrame, np.ndarray]:
    # Correct for z-step dependence. Assumes that impact is the same for all cycles
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    correct_z_only = 0
    logging.info(f'correct_fiducial: {fiducial_name}')
    min_cycle, max_cycle = map(int, config['cycle_range'].split('-'))
    min_frame, max_frame = map(int, config['frame_range'].split('-'))
    min_z_step, max_z_step = map(int, config['z_step_range'].split('-'))
    min_time_point, max_time_point = map(int, config['time_point_range'].split('-'))
    num_time_points = max_time_point - min_time_point + 1
    num_frames = max_frame - min_frame + 1
    num_cycles = max_cycle - min_cycle + 1
    num_z_steps = max_z_step - min_z_step + 1
    total_cycles = num_cycles * num_time_points
    frames_per_cycle = num_frames * num_z_steps
    dimnames = config['dimnames']
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    ndims = len(xyz_colnames)
    sd_colnames = [config['x_sd_col'], config['y_sd_col'], config['z_sd_col']]
    # Create array to hold x,y,z for each cycle
    x_ct = np.zeros((total_cycles, frames_per_cycle), dtype=float)
    sd_ct = np.zeros((total_cycles, frames_per_cycle), dtype=float)
    x_ct.fill(np.nan)
    for k in range(ndims):
        # Skip x and y
        if correct_z_only and k<2:
            return df
        colname = xyz_colnames[k]
        logging.info(f'Correcting #{fiducial_label} label:{fiducial_name}:{colname}')
        # Get x,y,z values for each cycle
        for j in range(num_time_points):
            for i in range(num_cycles):
                idx = (
                    (df[config['cycle_col']] == i + min_cycle) &
                    (df[config['time_point_col']] == j + min_time_point)
                )
                df_sel = df[idx]
                cycle_index = i + j * num_cycles
                frame_index = df_sel[config['frame_col']] - min_frame + (df_sel[config['z_step_col']] - min_z_step) * num_frames
                x_ct[cycle_index, frame_index] = df_sel[colname].values
                sd_ct[cycle_index, frame_index] = df_sel[sd_colnames[k]].values
        # At this point x_ct and sd_ct contain the x,y,z values for each cycle/timepoint
        # x_ct has NAs for missing frames and sd_ct has zeros for missing frames
        dx_c = make_corrections_for_cycles(x_ct, sd_ct, config)
        c_z_step = make_corrections_for_zstep(x_ct, sd_ct, dx_c, config)
        x_ct_cor = apply_corrections_for_zstep(x_ct, c_z_step, config)
        sd_t = estimate_errors_for_zstep(x_ct_cor, config)
        # Transfer corrected values back to df
        for j in range(num_time_points):
            for i in range(num_cycles):
                idx = (
                        (df[config['cycle_col']] == i + min_cycle) &
                    (df[config['time_point_col']] == j + min_time_point)
                )
                cycle_index = i + j * num_cycles
                frame_index = df[idx][config['frame_col']] - min_frame + (df[idx][config['z_step_col']] - min_z_step) * num_frames
                df.loc[idx, colname] = x_ct_cor[cycle_index, frame_index]
                # Transfer corrected sd values back to df
                # For those elements without sd value from correction, use the old value
                # This can happen because there weren't enough detections to estimate the error
                non_nan_mask = ~np.isnan(sd_t[frame_index]) & (sd_t[frame_index] != 0)
                df.loc[idx, sd_colnames[k]] = np.where(non_nan_mask, sd_t[frame_index], df.loc[idx, sd_colnames[k]])

        dim = dimnames[k]
        outdir = os.path.join(config['fiducial_dir'], f"{fiducial_name}")
        outpath = os.path.join(outdir, f"{fiducial_name}_cor_{dim}_vs_frame")
        plot_scatter(df[config['image_id_col']], df[colname], 'image-ID', f'{dim} (nm)', f"{dim} corrected for z-step vs frame",
                     outpath, config)
        plotly_scatter(df[config['image_id_col']], df[colname], df[sd_colnames[k]], 'image-ID', f'{dim} (nm)', f"{dim} corrected for z-step vs frame",
                       outpath, config)
        # Plot fitted values on top of original values
        if config['plot_per_fiducial_fitting']:
            plot_fiduciual_zstep_fit(fiducial_label,df,dim, config)
    return df, idx_cor

def plot_fiduciual_zstep_fit(fiducial_index: int, df: pd.DataFrame,dim: str, config: dict) -> int:
    logging.info('plot_fiduciual_zstep_fit')
    # Plots the original and corrected values for a particular dim estimate of a fiducial
    outdir = os.path.join(config['output_dir'], "fiducial_zstep_fit")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"f_{fiducial_index}_d_{dim}_fit")
    x = df[config['image_id_col']]
    y_cor_col = config[f'{dim}_col']
    y_cor_sd_col = config[f'{dim}_sd_col']
    y_orig_col = f'{y_cor_col}_0'
    y_orig_sd_col = f'{y_cor_sd_col}_0'
    y_cor = df[y_cor_col]
    y_cor_sd = df[y_cor_sd_col]
    y_orig = df[y_orig_col]
    y_orig_sd = df[y_orig_sd_col]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    sc1 = ax1.scatter(x, y_orig, s=0.1, c=y_orig_sd, label='Original')
    # Add sd of y_orig to the plot
    sd_y = np.nanstd(y_orig)
    ax1.text(0.1, 0.9, f'sd = {sd_y:.2f}', transform=ax1.transAxes)
    ax1.set_xlabel('image-ID')
    ax1.set_ylabel(f"{dim} (nm)")
    ax1.set_title(f'Original {dim} fit for fid={fiducial_index}')
    ax1.legend()
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Original SD')

    sc2 = ax2.scatter(x, y_cor, s=0.1, c=y_cor_sd, label='Corrected')
    # Add sd of y_orig to the plot
    sd_y = np.nanstd(y_cor)
    ax2.text(0.1, 0.9, f'sd = {sd_y:.2f}', transform=ax2.transAxes)
    ax2.set_xlabel('image-ID')
    ax2.set_ylabel(f"{dim} (nm)")
    ax2.set_title(f'Corrected {dim} fit for fid={fiducial_index}')
    ax2.legend()
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Corrected SD')

    # Determine the combined y-range
    y_min = min(y_orig.min(), y_cor.min())
    y_max = max(y_orig.max(), y_cor.max())
    # Set the same y-range for both axes
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return 0

def make_corrections_for_cycles(x_ct: np.ndarray, sd_ct: np.ndarray, config: dict) -> np.ndarray:
    # Make corrections for cycles - add this number to each cycle of fiducial.
    # Meant to correct for drift during z-step compensation.
    # There may be nan's since not all fiducials have values at all frames
    # Fill in unknown values with zero
    logging.info('make_corrections_for_cycles')
    x_ct_masked = np.ma.masked_invalid(x_ct)
    sd_ct_masked = np.ma.masked_invalid(sd_ct)
    combined_mask = np.logical_or(x_ct_masked.mask, sd_ct_masked.mask)
    non_zero_mask = sd_ct != 0
    weights = np.full_like(sd_ct, np.nan, dtype=float)
    weights[non_zero_mask] = 1 / sd_ct[non_zero_mask]**2
    x_ct_masked_combined = np.ma.masked_array(x_ct, mask=combined_mask)
    weights_masked_combined = np.ma.masked_array(weights, mask=combined_mask)
    # Weighted average, ignoring NaNs
    c_cycle = -np.ma.average(x_ct_masked_combined, axis=1, weights=weights_masked_combined).filled(0)
    return c_cycle[:, None]

def make_corrections_for_zstep(x_ct: np.ndarray, sd_ct: np.ndarray, dx_c: np.ndarray, config: dict) -> np.ndarray:
    # Make corrections for zstep
    # There may be nan's since not all fiducials have values at all frames
    # Fill in unknown values with zeros
    # The correction is just the weighted average of the fiducials after correction for cycles (drift and offset)
    logging.info('make_corrections_for_zstep')
    min_z_step, max_z_step = map(int, config['z_step_range'].split('-'))
    num_zsteps = max_z_step - min_z_step + 1
    total_frames = x_ct.shape[1]
    frames_per_zstep = int(total_frames / num_zsteps)
    c_z_step = np.zeros(num_zsteps)
    for i in range(num_zsteps):
        x_ct_z = x_ct[:, i * frames_per_zstep:(i + 1) * frames_per_zstep] + dx_c
        sd_ct_z = sd_ct[:, i * frames_per_zstep:(i + 1) * frames_per_zstep]
        x_ct_masked = np.ma.masked_invalid(x_ct_z)
        sd_ct_masked = np.ma.masked_invalid(sd_ct_z)
        combined_mask = np.logical_or(x_ct_masked.mask, sd_ct_masked.mask)
        non_zero_mask = sd_ct_z != 0
        weights = np.full_like(sd_ct_z, np.nan, dtype=float)
        weights[non_zero_mask] = 1 / sd_ct_z[non_zero_mask]**2
        x_ct_masked_combined = np.ma.masked_array(x_ct_z, mask=combined_mask)
        weights_masked_combined = np.ma.masked_array(weights, mask=combined_mask)
        # Weighted average, ignoring NaNs
        if np.any(~x_ct_masked_combined.mask):
            c_z_step[i] = -np.ma.average(x_ct_masked_combined, weights=weights_masked_combined)
    return c_z_step

def apply_corrections_for_cycles(x_ct: np.ndarray, dx_c: np.ndarray, config: dict) -> np.ndarray:
    # Apply corrections for cycles
    logging.info('apply_corrections_for_cycles')
    x_ct_cor = x_ct + dx_c
    return x_ct_cor

def apply_corrections_for_zstep(x_ct: np.ndarray, c_z_step: np.ndarray, config: dict) -> np.ndarray:
    num_zsteps = c_z_step.shape[0]
    total_frames = x_ct.shape[1]
    frames_per_zstep = int(total_frames / num_zsteps)

    ct_z_step = np.zeros_like(x_ct)
    for i in range(num_zsteps):
        ct_z_step[:, i * frames_per_zstep:(i + 1) * frames_per_zstep] = c_z_step[i]
    x_ct_cor = x_ct + ct_z_step
    return x_ct_cor

def estimate_errors_for_zstep(x_ct: np.ndarray, config: dict) -> np.ndarray:
    logging.info('estimate_errors_for_zstep')
    valid_counts = np.sum(~np.isnan(x_ct), axis=0)
    idx = valid_counts > 2 # Need at least 3 values to estimate sd and avoid tiny values
    sd_t = np.full(x_ct.shape[1], np.nan)
    sd_t[idx] = np.nanstd(x_ct[:, idx], axis=0)
    return sd_t

def zstep_correction_cost_function(c_z_step: np.ndarray, x_ct: np.ndarray, sd_ct: np.ndarray, dx_c, config: dict) -> float:
    # Calculate the cost function for the corrections
    x_ct_cor = apply_corrections_for_zstep(x_ct, c_z_step, config) + dx_c
    # sd_t = estimate_errors_for_zstep(x_ct_cor, config)
    cost = np.nansum(x_ct_cor**2 / sd_ct**2)
    # print(f'Cost: {cost} sum(c_z_step): {np.sum(c_z_step)}')
    return cost

def plot_fitted_fiducial(fiducial_name, j, k, x_fit_ft, xsd_fit_ft, config):
    outdir = os.path.join(config['fiducial_dir'], f"{fiducial_name}")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{fiducial_name}_{config['dimnames'][k]}_fit_vs_frame")
    plotly_scatter(np.arange(x_fit_ft.shape[2]), x_fit_ft[k, j, :], None,
                   'image-ID', f'{config["dimnames"][k]} (nm)', f'{config["dimnames"][k]} fit vs frame', outpath, config)

def plot_fitted_fiducials_parallel(df_fiducials: pd.DataFrame, x_fit_ft: np.ndarray, xsd_fit_ft: np.ndarray, config: dict) -> int:
    logging.info('plot_fitted_fiducials_parallel')

    ndims = x_fit_ft.shape[0]
    nfiducials = x_fit_ft.shape[1]
    tasks = [(df_fiducials.name[j], j, k, x_fit_ft, xsd_fit_ft, config) for j in range(nfiducials) for k in range(ndims)]

    with multiprocessing.Pool() as pool:
        pool.starmap(plot_fitted_fiducial, tasks)

    return 0

def optimise_dim(j: int, nfiducials: int, x_ret: np.ndarray, w: np.ndarray) -> np.ndarray:
    dimensions = ['x', 'y', 'z']
    logging.info(f'Grouping fiducials fits to {dimensions[j]}')
    optimise_method = "L-BFGS-B" # L-BFGS-B, Powell, CG, BFGS, Nelder-Mead, TNC, COBYLA, SLSQP, Newton-CG, trust-ncg, trust-krylov, trust-exact, trust-constr
    # TODO: Put in better initial guess, add bounds and Jacobian function
    initial_offsets = np.zeros(nfiducials - 1)
    initial_cost = variance_cost(initial_offsets, x_ret[j,:,:], w[j,:,:])
    result = scipy.optimize.minimize(variance_cost, initial_offsets, args=(x_ret[j,:,:], w[j,:,:]),
                                    method=optimise_method,options = {'maxiter': 1e5, 'disp': True})
    optimal_offsets = result.x
    final_cost = result.fun
    logging.info(f'Initial cost: {initial_cost} Final cost: {final_cost}')
    x_ret[j,:,:] = apply_offsets(optimal_offsets, x_ret[j,:,:])
    return x_ret[j,:,:]

def variance_cost(offsets, x, w):
    # x_shifted = apply_offsets(offsets, x)
    nfiducials = x.shape[0]
    x_shifted = x[1:nfiducials, :] + offsets[:nfiducials-1, np.newaxis]
    # At each time point, calculate the variance of the fiducial fits across time points
    # Weight this by the uncertainties at each time point
    var_t = np.nanvar(x_shifted, axis=0)
    weight_t = np.sum(w, axis=0)
    ret = np.sum(var_t * weight_t)
    # logging.info(f'Cost: {ret} offsets: {offsets}')
    return ret

def apply_offsets(offsets, x):
    nfiducials = x.shape[0]
    x_shifted = x.copy()
    # Apply offset to all fiducials except the first
    x_shifted[1:nfiducials, :] += offsets[:nfiducials-1, np.newaxis]
    return x_shifted

def minimize_fiducial_fit_variance_parallel(x_ft: np.ndarray, xsd_ft: np.ndarray,config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('minimize_fiducial_fit_variance_parallel')
    # Finds the offsets that minimise the variance of the fiducial fits at each time point
    nfiducials = x_ft.shape[1]
    ndimensions = x_ft.shape[0]
    x_ret = x_ft.copy()
    w = 1 / xsd_ft**2 # weight to be used in the cost function

    with multiprocessing.Pool() as pool:
        results = pool.starmap(optimise_dim, [(j, nfiducials, x_ret, w) for j in range(ndimensions)])

    for j in range(ndimensions):
        x_ret[j,:,:] = results[j]

    return x_ret, xsd_ft


def plot_fiduciual_step_fit(fiducial_index: int, interval_index: int, dimension_index: int, y: np.ndarray, ysd: np.ndarray, y_fit: np.ndarray, ysd_fit: np.ndarray, config: dict) -> int:
    logging.info('plot_fiduciual_step_fit')
    dimnames = config['dimnames']
    dim = dimnames[dimension_index]
    outdir = os.path.join(config['output_dir'], "fiducial_step_fit")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"fidx_{fiducial_index}_i_{interval_index}_d_{dim}_fit")
    x = np.arange(len(y))
    plt.figure(figsize=(10, 6))
    if np.sum(~np.isnan(y)) == 0 or np.sum(~np.isnan(ysd)) == 0 or np.sum(~np.isnan(y_fit)) == 0:
        logging.warning(f'No valid data for fitting in plot_fiduciual_step_fit() for fiducial {fiducial_index} dimension {dim} interval {interval_index}')
        return
    sc = plt.scatter(x, y, c = ysd, s = 0.1, label='Original Data')
    plt.colorbar(sc, label='sd')
    plt.scatter(x, y_fit+ysd_fit, s=0.1, label='fit+sd')
    plt.scatter(x, y_fit-ysd_fit, s=0.1, label='fit-sd')
    plt.scatter(x, y_fit, s=0.1, label='fit')
    plt.xlabel('image-ID')
    plt.ylabel(f"{dim} (nm)")
    plt.title(f'Fit for {dim}  fid={fiducial_index} tp={interval_index}')
    plt.legend()
    plt.savefig(outpath)
    plt.close()
    return 0

def fit_fiducial_step(xt: np.ndarray, xt_sd: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    polynomial_degree = config['polynomial_degree']
    use_weights_in_fit = (config['use_weights_in_fit']!=0)
    extrapolate_to_end = True
    median_filter_size = 100
    outlier_threshold = 3

    non_nan_indices = ~np.isnan(xt) & ~np.isnan(xt_sd) & (xt_sd!=0)
    nan_indices = ~non_nan_indices
    if np.sum(non_nan_indices) == 0:
        logging.warning('No valid data for fitting in fit_fiducial_step()')
        return np.full_like(xt, np.nan), np.full_like(xt, np.nan)

    first_non_nan = np.min(np.where(non_nan_indices))
    last_non_nan = np.max(np.where(non_nan_indices))
    x = np.arange(len(xt))
    y = xt

    # Fit a polynomial using weights
    if use_weights_in_fit:
        w = 1 / xt_sd
        coefficients = np.polyfit(x[non_nan_indices], y[non_nan_indices], polynomial_degree, w=w[non_nan_indices])
    else:
        coefficients = np.polyfit(x[non_nan_indices], y[non_nan_indices], polynomial_degree)
    x_fit = np.polyval(coefficients, x)

    # Calculate error bars of the fit for later use when weighting fits
    residuals = np.abs(y - x_fit)
    outlier_sd = np.nanmean(residuals[non_nan_indices]) * outlier_threshold
    residuals_filled = np.where(nan_indices, outlier_sd, residuals)
    smoothed_residuals = scipy.ndimage.median_filter(residuals_filled, size=median_filter_size)
    xsd_fit = np.copy(smoothed_residuals)
    xsd_fit[xsd_fit>outlier_threshold | np.isnan(residuals)] = outlier_sd

    if not extrapolate_to_end:
        if first_non_nan > 0:
            x_fit[:first_non_nan] = x_fit[first_non_nan]
            xsd_fit[:first_non_nan] = np.nanmean(xsd_fit[non_nan_indices]) * 3
        if last_non_nan < len(x_fit):
            x_fit[last_non_nan:] = x_fit[last_non_nan]
            xsd_fit[last_non_nan:] = np.nanmean(xsd_fit[non_nan_indices]) * 3

    return x_fit, xsd_fit

