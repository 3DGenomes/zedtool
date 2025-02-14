#!/usr/bin/env python3
import numpy as np
import matplotlib
import os
import yaml
import sys
import logging
import pandas as pd
import shutil
from zedtool.detections import filter_detections, mask_detections, bin_detections, bins3d_to_stats2d, make_density_mask_2d, make_image_index, create_backup_columns
from zedtool.plots import plot_detections, plot_binned_detections_stats, plot_fiducials, plot_summary_stats, plot_fiducial_quality_metrics, save_to_tiff_3d
from zedtool.fiducials import find_fiducials, make_fiducial_stats, filter_fiducials, correct_fiducials, plot_fiducial_correlations, make_quality_metrics, correct_detections, apply_corrections, compute_deltaz
from zedtool.configuration import config_validate, config_update, config_validate_detections, config_default, config_print
from zedtool import __version__


# Prints some debugging plots for an SRX dataset.
# Write out a table with both corrected and uncorrected z.

def main(yaml_config_file: str) -> int:
    no_display = True
    # no_display = False
    # Check if running in headless mode
    if os.getenv('DISPLAY') is None or os.getenv('SLURM_JOBID') is not None or no_display == True:
        matplotlib.use('agg')  # Use the 'agg' backend for headless mode
    else:
        matplotlib.use('TkAgg')  # Use the 'TkAgg' backend if a display is available
    print_ascii_logo()
    print_version()
    print(f"Reading config file: {yaml_config_file}")
    # read yaml config file
    with open(yaml_config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return 1
    config = config_update(config_default(), config) # Update defaults with settings from file
    debug = config['debug']
    # set up logging
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
    # quieten matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    matplotlib.set_loglevel("error")
    logging.getLogger('matplotlib').propagate = False

    config['fiducial_dir'] = os.path.join(config['output_dir'], 'fiducials')
    detections_file = config['detections_file']
    binary_detections_file = os.path.join(config['output_dir'],config['binary_detections_file'])

    noclobber = config['noclobber']
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['fiducial_dir'], exist_ok=True)
    if not config_validate(config):
        return 1
    if debug:
        config_print(config)

    if noclobber and os.path.exists(binary_detections_file) and config['make_caches']:
        logging.info(f"Loading detections from {binary_detections_file}")
        df = pd.read_pickle(binary_detections_file)
    else:
        logging.info(f"Reading detections from {detections_file}")
        df = pd.read_csv(detections_file)
        if config['make_caches']:
            df.to_pickle(binary_detections_file)

    logging.info(f"Loaded {df.shape[0]} rows")
    if not config_validate_detections(df, config):
        return 1

    if config['apply_drift_correction']:
        drift_correction_file = config['drift_correction_file']
        logging.info("Applying drift correction from {drift_correction_file")
        # read corrections from tsv file
        df_corrections = pd.read_csv(drift_correction_file, sep='\t')
        x_t = np.zeros((3,df_corrections.shape[0]))
        x_t[0] = df_corrections['x'].values
        x_t[1] = df_corrections['y'].values
        x_t[2] = df_corrections['z'].values
        df = apply_corrections(df, x_t, config)

    df = compute_deltaz(df, config) # add deltaz column
    df = filter_detections(df, config)
    logging.info(f"Filtered to {df.shape[0]} rows")

    # Combine x,y,x into a single array with shape (n,3)
    det_xyz = np.vstack((df[config['x_col']].values, df[config['y_col']].values, df[config['z_col']].values)).T
    bin_resolution = config['bin_resolution']
    counts_xyz, x_bins, y_bins, z_bins = bin_detections(det_xyz,bin_resolution)

    # Calculate moments and variance
    n_xy, mean_xy, sd_xy = bins3d_to_stats2d(counts_xyz, z_bins)

    # Plot detections before masking
    if config['plot_detections']:
        plot_detections(df,'detections_summary', config)
        plot_binned_detections_stats(n_xy, mean_xy, sd_xy, 'binned_detections_summary',config)
        save_to_tiff_3d(counts_xyz,"binned_detections", config)

    # Make index into the binned xy image from the detections
    x_idx, y_idx, z_idx = make_image_index(det_xyz, x_bins, y_bins, z_bins)

    if config['mask_on_density']:
        # Mask on density to remove bright/dim areas
        # Mostly unused but can speed up processing and remove background
        mask_xy = make_density_mask_2d(n_xy, config)
        logging.info(f"Before masking: {np.sum(mask_xy)} detections")
        # Select detections in mask_xy
        idx = mask_detections(mask_xy, x_idx, y_idx)
        logging.info(f"After masking: {np.sum(idx)} detections")
        # Apply masks
        det_xyz = det_xyz[idx, :]
        df = df[idx]
        x_idx = x_idx[idx]
        y_idx = y_idx[idx]
        z_idx = z_idx[idx]

    # Find fiducials
    # Treat n_xy as an image and segment, expand segmented areas, make labels and attached to df. Save centroids and labels
    df, df_fiducials = find_fiducials(n_xy, df, x_idx, y_idx, config)

    # Find wobbliness, detections per fiducial, correlation between x, y and z for each fiducial
    df_fiducials = make_fiducial_stats(df_fiducials, df, config)
    outpath = os.path.join(config['fiducial_dir'], "fiducials_unfiltered.tsv")
    df_fiducials.to_csv(outpath, sep='\t', index=False)

    # Remove problematic and outlier fiducials
    df_filtered_fiducial_detections, df_fiducials = filter_fiducials(df_fiducials, df, config)
    outpath = os.path.join(config['fiducial_dir'], "fiducials_filtered.tsv")
    df_fiducials.to_csv(outpath, sep='\t', index=False)
    outpath = os.path.join(config['fiducial_dir'], "fiducials_detections_filtered.tsv")
    df_filtered_fiducial_detections.to_csv(outpath, sep = '\t', index=False)

    if config['make_quality_metrics']:
        df_metrics = make_quality_metrics(df, df_fiducials, config)
        outpath = os.path.join(config['output_dir'], "quality_metrics_before_correction.tsv")
        df_metrics.to_csv(outpath, sep='\t', index=False)
        # Only plot before correction
        plot_fiducial_quality_metrics(df_fiducials, config)

    # Make correlations between fiducials between and within sweeps
    if config['plot_fiducial_correlations']:
        plot_fiducial_correlations(df_fiducials, df_filtered_fiducial_detections, config)

    if config['plot_fiducials']:
        plot_fiducials(df_fiducials, df_filtered_fiducial_detections, config)

    if config['plot_summary_stats']:
        plot_summary_stats(df, det_xyz, config)

    # Backup x,y,z, and sd columns in df to x1, y1, z1,... before they are changed
    if config['correct_fiducials'] or config['correct_detections']:
        df = create_backup_columns(df, config)

    # Correct fiducials with zstep model
    if config['correct_fiducials']:
        df_fiducials, df = correct_fiducials(df_fiducials, df, config)

    # Correct detections using (possibly corrected) fiducials
    if config['correct_detections']:
        df = correct_detections(df, df_fiducials, config)

    if config['make_quality_metrics']:
        df_fiducials = make_fiducial_stats(df_fiducials, df, config)
        outpath = os.path.join(config['fiducial_dir'], "fiducials_corrected.tsv")
        df_fiducials.to_csv(outpath, sep='\t', index=False)
        df_metrics = make_quality_metrics(df, df_fiducials, config)
        outpath = os.path.join(config['output_dir'], "quality_metrics_corrected.tsv")
        df_metrics.to_csv(outpath, sep='\t', index=False)

    # Write df and copy config file to output dir
    df = compute_deltaz(df, config) # update deltaz column
    df.to_csv(os.path.join(config['output_dir'], 'corrected_detections.csv'), index=False)
    shutil.copy(yaml_config_file, os.path.join(config['output_dir'], 'config.yaml'))

    return 0

def print_version() -> str:
    ret = f"  Version: {__version__}\n"
    print(ret)
    return ret

def print_ascii_logo() -> str:
    ret = """
 +-+-+-+-+-+-+-+
 |z|e|d|t|o|o|l|
 +-+-+-+-+-+-+-+
     """
    print(ret)
    return ret

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: zedtool.py <config_file>")
        print_version()
        sys.exit(1)
    ret = main(sys.argv[1])
    print(ret)
    sys.exit(ret)
