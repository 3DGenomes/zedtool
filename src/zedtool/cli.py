#!/usr/bin/env python3
import numpy as np
import matplotlib
import os
import yaml
import sys
import logging
import pandas as pd
from zedtool.detections import filter_detections, mask_detections, bin_detections, bins3d_to_stats2d, make_density_mask_2d, make_image_index, reorder_sweeps
from zedtool.plots import plot_scatter, plot_histogram, plot_detections_summary, plot_binned_detections_summary, plot_fiducials, plotly_scatter
from zedtool.srxstats import extract_z_correction, z_means_by_marker
from zedtool.fiducials import find_fiducials, make_fiducial_stats, filter_fiducials, correct_fiducials, make_fiducial_correlations

from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple

# Prints some debugging plots for an SRX dataset.
# Takes a corrected and an uncorrected table of detections, registers the rows and finds the corrections.
# Write out a table with both corrected and uncorrected z.



def main(yaml_config_file: str) -> int:
    no_display = True
    # no_display = False
    # Check if running in headless mode
    if os.getenv('DISPLAY') is None or os.getenv('SLURM_JOBID') is not None or no_display == True:
        matplotlib.use('agg')  # Use the 'agg' backend for headless mode
    else:
        matplotlib.use('TkAgg')  # Use the 'TkAgg' backend if a display is available

    print(f"Reading config file: {yaml_config_file}")
    # read yaml config file
    with open(yaml_config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # set up logging
    logging.basicConfig(level=logging.DEBUG if config['debug'] else logging.INFO)
    logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    # quieten matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    config['fiducial_dir'] = os.path.join(config['output_dir'], 'fiducials')
    detections_file = config['corrected_detections_file']
    binary_detections_file = os.path.join(config['output_dir'],config['binary_detections_file'])
    corrected_detections_with_original_file = os.path.join(config['output_dir'], "corrected_detections_with_original_z.csv")
    corrections_file = os.path.join(config['output_dir'], "tdz.tsv")
    plot_detections_summaries = config['plot_detections_summaries']

    debug = config['debug']
    have_correction = False
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['fiducial_dir'], exist_ok=True)
    z_step_step = config['z_step_step']

    if debug and os.path.exists(binary_detections_file) and config['make_caches']:
        logging.info(f"Loading detections from {binary_detections_file}")
        df = pd.read_pickle(binary_detections_file)
    else:
        logging.info(f"Reading detections from {detections_file}")
        df = pd.read_csv(detections_file)
        df = filter_detections(df, config)
        if config['debug']:
            df.to_pickle(binary_detections_file)

    logging.info(f"Loaded {df.shape[0]} rows")

    # if we have uncorrected_detections_file then we need to extract z-corrections
    if 'uncorrected_detections_file' in config:
        have_correction = True
        if not os.path.exists(corrected_detections_with_original_file):
            logging.info(f"Reading uncorrected detections from {config['uncorrected_detections_file']}")
            df_orig = pd.read_csv(config['uncorrected_detections_file'])
            df_orig = filter_detections(df_orig, config)
            df_tdz,df_cor = extract_z_correction(df, df_orig)
            df_cor.to_csv(f"{config['output_dir']}/corrected_detections_with_original_z.csv", index=False)
            df_tdz.to_csv(corrections_file, index=False, sep='\t')
            df_cor_orig_z = pd.read_csv(f"{config['output_dir']}/corrected_detections_with_original_z.csv")
        else:
            logging.info(f"Loading z-corrections from {corrections_file}")
            df_tdz = pd.read_csv(corrections_file, sep='\t')
            df_cor_orig_z = pd.read_csv(f"{config['output_dir']}/corrected_detections_with_original_z.csv")
        tdz_srx = df_tdz.values
    else:
        tdz_srx = np.zeros((1,2))
        df_cor_orig_z = df

    # Combine x,y,x into a single array with shape (n,3)
    det_xyz = np.vstack((df[config['x_col']].values, df[config['y_col']].values, df[config['z_col']].values)).T
    bin_resolution = config['bin_resolution']
    counts_xyz, x_bins, y_bins, z_bins = bin_detections(det_xyz,bin_resolution)


    # Calculate moments and variance
    n_xy, mean_xy, sd_xy = bins3d_to_stats2d(counts_xyz, z_bins)

    # Plot detections before masking
    if plot_detections_summaries:
        plot_detections_summary(df,'detections_summary', config)
        plot_binned_detections_summary(n_xy, mean_xy, sd_xy, 'binned_detections_summary',config)
        # plot_histogram(df[config['z_step_col']], 'z-step', 'Detections', "Detections by z-step", "zstep_histogram", df['z-step'].max()-df['z-step'].min()+1, config)
        plot_histogram(df[config['z_step_col']], 'z-step', 'Detections', "Detections by z-step", "zstep_histogram", config)


    # Make index into the binned xy image from the detections
    x_idx, y_idx, z_idx = make_image_index(det_xyz, x_bins, y_bins, z_bins)

    # # Mask on density to remove bright/dim areas
    # # Mostly unused but can speed up processing and remove background
    # mask_xy = make_density_mask_2d(n_xy, config)
    # logging.info(f"Before masking: {np.sum(mask_xy)} detections")
    # # Select detections in mask_xy
    # idx = mask_detections(mask_xy, x_idx, y_idx)
    # logging.info(f"After masking: {np.sum(idx)} detections")
    # # Apply masks
    # det_xyz = det_xyz[idx, :]
    # df = df[idx]
    # x_idx = x_idx[idx]
    # y_idx = y_idx[idx]
    # z_idx = z_idx[idx]

    # Find fiducials
    # Treat n_xy as an image and segment, expand segmented areas, make labels and attached to df. Save centroids and labels
    df, df_fiducials = find_fiducials(n_xy, df, x_idx, y_idx, config)

    # Find wobbliness, detections per fiducial, correlation between x, y and z for each fiducial
    df_fiducials = make_fiducial_stats(df_fiducials, df, config)

    # Remove problematic and outlier fiducials
    df_filtered_fiducials, df_fiducials = filter_fiducials(df_fiducials, df, config)

    # Make correlations between fiducials between and within sweeps
    make_fiducial_correlations(df_fiducials, df_filtered_fiducials, config)

    # Plot fiducials
    #   * plot z vs time, projections coloured by quantities, dendrogram of groupings
    plot_fiducials(df_fiducials, df_filtered_fiducials, config)


    # Correct fiducials with zstep model
    # df_fiducials, df = correct_fiducials(df_fiducials, df, config)

    # Re-order sweeps
    # df = reorder_sweeps(df, config)

    # TODO: Put these into a function and make it optional
    # Plot detections and other quantities
    plot_histogram(df[config['z_col']] - df[config['z_step_col']] * z_step_step, 'z - z-step*z_step_step', 'Detections','Detections by z-z-step*z_step_step', 'z_zstep_histogram', config)
    plot_scatter(df[config['frame_col']], df[config['z_col']], 'frame', 'z', 'z vs frame', 'z_vs_frame', config)
    plot_scatter(df[config['frame_col']], df[config['photons_col']], 'frame', 'photon-count','photon-count vs frame', 'photon_count_vs_frame', config)
    plot_scatter(df[config['frame_col']], df[config['z_step_col']], 'frame', 'z-step','z-step vs frame', 'zstep_vs_frame', config)
    plot_scatter(df[config['frame_col']], df[config['z_col']] - df[config['z_step_col']] * z_step_step, 'frame', 'z - z-step*z_step_step','z - z-step*z_step_step vs frame', 'z_zstep_vs_frame', config)

    z_mean,t = z_means_by_marker(det_xyz,df[config['frame_col']].values)
    plot_scatter(t, z_mean,'time','mean(z) per frame','mean(z) vs time','z_mean_per_frame_vs_time', config)

    z_mean,cycle = z_means_by_marker(det_xyz,df[config['cycle_col']].values)
    plot_scatter(cycle,z_mean,'cycle','mean(z) per cycle','mean(z) vs cycle','z_mean_per_cycle_vs_cycle', config)

    z_mean,z_step = z_means_by_marker(det_xyz,df[config['z_step_col']].values)
    plot_scatter(z_step, z_mean, 'z-step', 'mean(z) per z-step', 'mean(z) vs z-step', 'z_mean_per_z_step_vs_z_step', config)

    if have_correction:
        # Plot and compare tdz_srx with the above two
        plot_scatter(tdz_srx[:, 0], tdz_srx[:, 1], 'frame', 'dz(nm)', 'dz(nm) relative to start', 'dz_vs_frame', config)
    return 0

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: zedtools.py <config_file>")
        sys.exit(1)
    ret = main(sys.argv[1])
    print(ret)
    sys.exit(ret)
