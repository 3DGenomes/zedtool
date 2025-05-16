#!/usr/bin/env python3
import numpy as np
import matplotlib
import os
import yaml
import sys
import logging
import pandas as pd
import fsspec
from typing import Tuple
from zedtool.detections import filter_detections, mask_detections, bin_detections, bins3d_to_stats2d, make_density_mask_2d, make_image_index, create_backup_columns, compute_deltaz, compute_image_id, apply_corrections, deltaz_correct_detections, cat_experiment
from zedtool.plots import plot_detections, plot_binned_detections_stats, plot_fiducials, plot_summary_stats, plot_fiducial_quality_metrics, save_to_tiff_3d
from zedtool.fiducials import find_fiducials, make_fiducial_stats, filter_fiducials, zstep_correct_fiducials, plot_fiducial_correlations, make_quality_metrics, drift_correct_detections
from zedtool.configuration import config_validate, config_update, config_validate_detections, config_default, config_print
from zedtool.deconvolution import deconvolve_z
from zedtool.timepoints import plot_time_point_metrics
from zedtool import __version__

def read_config(yaml_config_file: str) -> dict:
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
            return {}

    # Set up config. This carries all the global variables and is not changed after this.
    config = config_update(config_default(), config) # Update defaults with settings from file
    config['fiducial_dir'] = os.path.join(config['output_dir'], 'fiducials')
    config['cache_dir'] = os.path.join(config['output_dir'],'cache')

    # Make output directories (requires config)
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['fiducial_dir'], exist_ok=True)
    if config['make_caches']:
        os.makedirs(config['cache_dir'], exist_ok=True)
    # set up logging (requires output_dir to exist)
    logfile = os.path.join(config['output_dir'], 'zedtool.log')
    print(f"Writing log output to {logfile}")
    logging.basicConfig(filename=logfile, level=logging.DEBUG if config['debug'] else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    # Add a StreamHandler to output logs to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if config['debug'] else logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)
    # quieten matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').propagate = False

    # Check config (requires logging)
    if not config_validate(config):
        logging.error("Config validation failed")
        return {}
    if config['debug']:
        config_print(config)

    # Print version
    logging.info(f"zedtool version: {__version__}")
    # If fiducials are included/excluded, convert the strings to lists
    if config['excluded_fiducials'] != None and isinstance(config['excluded_fiducials'], str) :
        config['excluded_fiducials'] = pd.Series(map(int, config['excluded_fiducials'].split(',')), dtype="int64")
        logging.info(f"Excluded fiducials: {config['excluded_fiducials']}")
    else:
        config['excluded_fiducials'] = pd.Series(dtype="int64")

    if config['included_fiducials'] != None and isinstance(config['included_fiducials'], str) :
        config['included_fiducials'] = pd.Series(map(int, config['included_fiducials'].split(',')), dtype="int64")
        logging.info(f"Included fiducials: {config['included_fiducials']}")
    else:
        config['included_fiducials'] = pd.Series(dtype="int64")

    # Parallel processing is incompatible with no_display = False
    if config['multiprocessing'] and no_display==False:
        logging.error("Parallel processing is not supported when not headless.")
        return {}

    # if num_threads is not set then use all available threads
    if config['num_threads'] is None:
        config['num_threads'] = os.cpu_count()
    if config['multiprocessing']:
        if config['num_threads'] > os.cpu_count():
            logging.warning(f"num_threads ({config['num_threads']}) is greater than available threads ({os.cpu_count()}). Using all available threads.")
            config['num_threads'] = os.cpu_count()
        logging.info(f"Using {config['num_threads']} threads for processing")
    else:
        logging.info("multiprocessing is disabled")
    return config

def read_detections(config: dict) -> pd.DataFrame:
    # read detections and cache if needed
    binary_detections_file = os.path.join(config['cache_dir'], 'detections.pkl')
    if config['noclobber'] and os.path.exists(binary_detections_file) and config['make_caches']:
        logging.info(f"Loading detections from {binary_detections_file}")
        df = pd.read_pickle(binary_detections_file)
    else:
        logging.info(f"Reading detections from {config['detections_file']}")
        with fsspec.open(config['detections_file']) as f:
            df = pd.read_csv(f)
        if config['make_caches']:
            df.to_pickle(binary_detections_file)

    # make sure detections are consistent with config
    logging.info(f"Loaded {df.shape[0]} rows")
    if not config_validate_detections(df, config):
        logging.error("Detections do not match config")
        return pd.DataFrame()
    return df

def pre_process_detections(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # processes that are only done once in the pipeline
    logging.info("pre_process_detections")
    # Optionally concatenate a second experiment - make sure this is the only action
    if config['concatenate_detections']:
        logging.info(f"Concatenating {config['concatenate_detections_file']}")
        df2 = pd.read_csv(config['concatenate_detections_file'])
        logging.info(f"Loaded {df2.shape[0]} rows to concatenate")
        if not config_validate_detections(df2, config):
            logging.error("Concatenated detections do not match config")
            return df
        df_offset = pd.read_csv(config['concatenate_offset_file'])
        df = cat_experiment(df, df2, df_offset, config)
        logging.info(f"Loaded {df.shape[0]} rows after concatenation")
        # Write out the concatenated file and return
        df.to_csv(os.path.join(config['output_dir'], 'concatenated_detections.csv'), index=False, float_format=config['float_format'])
        return df

    # Apply a pre-computed drift correction read from a file
    if config['apply_drift_correction']:
        drift_correction_file = config['drift_correction_file']
        logging.info(f"Applying drift correction from {drift_correction_file}")
        # read corrections from tsv file
        df_corrections = pd.read_csv(drift_correction_file, sep='\t')
        x_t = np.zeros((3,df_corrections.shape[0]))
        x_t[0] = df_corrections['x'].values
        x_t[1] = df_corrections['y'].values
        x_t[2] = df_corrections['z'].values
        df = apply_corrections(df, x_t, config)

    # Add any missing columnns
    df = compute_deltaz(df, config) # add deltaz column
    # Add image_id column if missing
    compute_image_id(df, config)
    # Remove detections outside of the selected columns' ranges
    if config['select_cols'] != '' and config['select_ranges'] != '':
        df = filter_detections(df, config)
    logging.info(f"Filtered to {df.shape[0]} rows")
    return df

def process_detections(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # processes that can be done multiple times in the pipeline
    # If this happens it will be with recursion - calling with modified config
    # and skipping the correction steps
    logging.info("process_detections")
    # Bin the detections to allow them to be treated as an image
    # Combine x,y,x into a single array with shape (n,3)
    det_xyz = np.vstack((df[config['x_col']].values, df[config['y_col']].values, df[config['z_col']].values)).T
    bin_resolution = config['bin_resolution']
    counts_xyz, x_bins, y_bins, z_bins = bin_detections(det_xyz,bin_resolution)

    # Calculate moments and variance
    n_xy, mean_xy, sd_xy = bins3d_to_stats2d(counts_xyz, z_bins)

    # Plot detections before masking
    if config['plot_detections']:
        # plot_detections() takes a _long_ time to run, so it's relegated to debug mode
        # binned_detections_summary.png will contain a lower resolution version of the same plot
        if config['debug']:
            plot_detections(df,'detections_summary', config)
        plot_binned_detections_stats(n_xy, mean_xy, sd_xy, 'binned_detections_summary',config)
        save_to_tiff_3d(counts_xyz,"binned_detections_3d", config)

    # Make index into the binned xyz image from the detections
    # x_idx gives the bin x-index for each detection, similarly for y and z
    x_idx, y_idx, z_idx = make_image_index(det_xyz, x_bins, y_bins, z_bins)

    if config['mask_on_density']:
        # Mask on density to remove bright/dim areas
        # Mostly unused but can speed up processing and remove background
        mask_xy = make_density_mask_2d(n_xy, config)
        logging.info(f"Before masking on density: {np.sum(mask_xy)} detections")
        # Select detections in mask_xy
        idx = mask_detections(mask_xy, x_idx, y_idx)
        logging.info(f"After masking on density: {np.sum(idx)} detections")
        # Apply masks
        det_xyz = det_xyz[idx, :]
        df = df[idx]
        x_idx = x_idx[idx]
        y_idx = y_idx[idx]
        z_idx = z_idx[idx]

    # Find fiducials in the binned image
    # Treat n_xy as an image and segment, expand segmented areas, make labels and attached to df. Save centroids and labels
    df, df_fiducials = find_fiducials(n_xy, df, x_idx, y_idx, config)

    # Find wobbliness, detections per fiducial, correlation between x, y and z for each fiducial
    df_fiducials = make_fiducial_stats(df_fiducials, df, config)
    outpath = os.path.join(config['fiducial_dir'], "fiducials_unfiltered.tsv")
    df_fiducials.to_csv(outpath, sep='\t', index=False, float_format=config['float_format'])

    # Remove problematic and outlier fiducials
    df, df_fiducials = filter_fiducials(df_fiducials, df, config)
    outpath = os.path.join(config['fiducial_dir'], "fiducials_filtered.tsv")
    df_fiducials.to_csv(outpath, sep='\t', index=False, float_format=config['float_format'])

    if config['make_quality_metrics']:
        df_metrics = make_quality_metrics(df, df_fiducials, config)
        outpath = os.path.join(config['output_dir'], "quality_metrics_summary.tsv")
        df_metrics.to_csv(outpath, sep='\t', index=False, float_format=config['float_format'])
        # Only plot before correction
        plot_fiducial_quality_metrics(df_fiducials, config)

    # Make correlations between fiducials between and within sweeps
    if config['plot_fiducial_correlations']:
        plot_fiducial_correlations(df_fiducials, df, config)

    if config['plot_fiducials']:
        plot_fiducials(df_fiducials, df, config)

    if config['plot_summary_stats']:
        plot_summary_stats(df, det_xyz, config)

    if config['plot_time_point_metrics']:
        plot_time_point_metrics(df_fiducials, df, config)

    making_corrrections = (config['zstep_correct_fiducials'] or
                           config['drift_correct_detections'] or
                           config['drift_correct_detections_multi_pass'] or
                            config['deltaz_correct_detections'] or
                            config['deconvolve_z']
                           )
    # Backup x,y,z, and sd columns in df to x1, y1, z1,... before they are changed
    if making_corrrections:
        df = create_backup_columns(df, config)

    # Correct fiducials with zstep model
    if config['zstep_correct_fiducials']:
        df_fiducials, df = zstep_correct_fiducials(df_fiducials, df, config)

    # Correct detections using (possibly corrected) fiducials
    # df_fiducials gains cols with consensus and fitting error
    if config['drift_correct_detections']:
        df, df_fiducials = drift_correct_detections(df, df_fiducials, config)
        # df_fiducials gains cols with consensus and fitting error, save this
        outpath = os.path.join(config['fiducial_dir'], "fiducials_drift_corrected.tsv")
        df_fiducials.to_csv(outpath, sep='\t', index=False, float_format=config['float_format'])

    if config['drift_correct_detections_multi_pass']:
        df, df_fiducials = drift_correct_detections_multi_pass(df_fiducials, df, config)

    if config['deltaz_correct_detections']:
        # redo fiducial to update regression against deltaz
        if config['drift_correct_detections']:
            df_fiducials = make_fiducial_stats(df_fiducials, df, config)
        df = deltaz_correct_detections(df, df_fiducials, config)

    if config['deconvolve_z']:
        df = deconvolve_z(df, df_fiducials, n_xy, x_idx, y_idx, config)

    # Write corrected df to output dir if it's been changed
    if making_corrrections:
        df = compute_deltaz(df, config) # update deltaz column in case you're saving it
        is_fiducial = df['is_fiducial'].values  #  column may be removed, but we still need it
        df = df[config['output_column_names']] # Select output_column_names from df
        output_file = os.path.join(config['output_dir'], 'corrected_detections.csv')
        logging.info(f"Writing corrected detections to {output_file}")
        df.to_csv(output_file, index=False, float_format=config['float_format'])
        # Save non-fiducials to csv
        if config['save_non_fiducial_detections']:
            output_file = os.path.join(config['output_dir'], 'corrected_detections_no_fiducials.csv')
            logging.info(f"Writing non-fiducial corrected detections to {output_file}")
            df[is_fiducial == 0].to_csv(output_file, index=False, float_format=config['float_format'])

    return df

def post_process_detections(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info("post_process_detections")

    making_corrrections = (config['zstep_correct_fiducials'] or
                           config['drift_correct_detections'] or
                           config['drift_correct_detections_multi_pass'] or
                            config['deltaz_correct_detections'] or
                            config['deconvolve_z']
                           )
    if not making_corrrections:
        # No corrections, no need to post process
        return df

    config_post = config.copy()
    config_post['detections_file'] = os.path.join(config['output_dir'], 'corrected_detections.csv')

    # don't do corrections
    config_post['drift_correct_detections'] = 0
    config_post['drift_correct_detections_multi_pass'] = 0
    config_post['zstep_correct_fiducials'] = 0
    config_post['deltaz_correct_detections'] = 0
    config_post['deconvolve_z'] = 0

    # don't make caches if we're just plotting the first pass
    config_post['make_caches'] = 0
    # make new directory for post processing outputs
    config_post['output_dir'] = f"{config_post['output_dir']}_corrected"
    config_post['fiducial_dir'] = os.path.join(config_post['output_dir'], 'fiducials')

    # don't use included/excluded fiducials because the labelling will have changed
    config_post['excluded_fiducials'] =  pd.Series(dtype="int64")
    config_post['included_fiducials'] =  pd.Series(dtype="int64")

    # Make output directories for second pass
    os.makedirs(config_post['output_dir'], exist_ok=True)
    os.makedirs(config_post['fiducial_dir'], exist_ok=True)
    # add deltaz column in case it was removed before saving corrected detections
    df = compute_deltaz(df, config)
    # add image_id column in case it was removed before saving corrected detections
    compute_image_id(df, config)
    # recursively call process_detections with the new config just to do the post-correction plotting
    df = process_detections(df, config_post)
    return df

def drift_correct_detections_multi_pass(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Corrects the drift of the detections using the fiducials
    # This is a multi-pass version of drift_correct_detections
    # It uses the fiducials to correct the detections and then uses the corrected detections to correct the fiducials
    logging.info("drift_correct_detections_multi_pass")

    # initial drift correction
    df, df_fiducials = drift_correct_detections(df, df_fiducials, config)

    outpath = os.path.join(config['fiducial_dir'], "fiducials_drift_corrected_1.tsv")
    df_fiducials.to_csv(outpath, sep='\t', index=False, float_format=config['float_format'])
    df_fiducials = make_fiducial_stats(df_fiducials, df, config)
    df_metrics = make_quality_metrics(df, df_fiducials, config)
    outpath = os.path.join(config['output_dir'], "quality_metrics_summary_pass_1.tsv")
    df_metrics.to_csv(outpath, sep='\t', index=False, float_format=config['float_format'])

    # redo drift correction after filtering already-corrected fiducials.
    df, df_fiducials = filter_fiducials(df_fiducials, df, config)
    df, df_fiducials = drift_correct_detections(df, df_fiducials, config)

    outpath = os.path.join(config['fiducial_dir'], "fiducials_drift_corrected_2.tsv")
    df_fiducials = make_fiducial_stats(df_fiducials, df, config)
    df_metrics = make_quality_metrics(df, df_fiducials, config)
    outpath = os.path.join(config['output_dir'], "quality_metrics_summary_pass_2.tsv")
    df_metrics.to_csv(outpath, sep='\t', index=False, float_format=config['float_format'])

    return df, df_fiducials

