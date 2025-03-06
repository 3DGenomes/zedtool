import logging
import os
import pandas as pd

def config_default() -> dict:
    config = {
        'detections_file': 'input.csv',
        'output_dir': 'output',
        'binary_detections_file': 'detections.npy',
        'drift_correction_file': 'drift_correction.tsv',
        'binned_detections_file': 'binned_detections.tif',
        'bin_resolution': 20,
        'z_step_step': -100,
        # Debugging and misc settings
        'debug': 0,
        'noclobber': 0,
        'make_caches': 0,
        'multiprocessing': 1,
        'float_format': '%.6g', #'%8g', # '%.2f',
        # Plot labels - globals, not from config file
        'dimnames': ['x','y','z'], # not from config file
        'timename': 't', # not from config file
        # Filtering settings
        'n_min_cutoff': 1,
        'n_max_cutoff': 10000000000,
        'select_cols': '',
        'select_ranges': '0-0',
        # experiment settings
        'frame_range': '0-0',
        'z_step_range': '0-0',
        'cycle_range': '0-0',
        'time_point_range': '0-0',
        # columns to read from detections file
        'frame_col': 'frame',
        'image_id_col': 'image-ID',
        'z_step_col': 'z-step',
        'cycle_col': 'cycle',
        'time_point_col': 'time-point',
        'x_col': 'x',
        'y_col': 'y',
        'z_col': 'z',
        'x_sd_col': 'precisionx',
        'y_sd_col': 'precisiony',
        'z_sd_col': 'precisionz',
        'photons_col': 'photon-count',
        'chisq_col': 'chisq',
        'deltaz_col': 'deltaz',
        'log_likelihood_col': 'log-likelihood',
        'llr_col': 'llr',
        'probe_col': 'vis',
        # fiducial settings
        'excluded_fiducials': None,
        'median_filter_disc_radius': 1,
        'filling_disc_radius': 10,
        'dilation_disc_radius': 10,
        'min_fiducial_size': 100,
        'min_fiducial_detections': 10,
        'max_detections_per_image': 1.1,
        'quantile_tail_cutoff': 0,
        'polynomial_degree': 2,
        'use_weights_in_fit': 0,
        'only_fiducials': 0,
        'consensus_method': 'median',
        # Deconvolution settings
        'decon_min_cluster_sd': 10,
        'decon_sd_shrink_ratio': 0.25,
        'decon_bin_threshold': 100,
        'decon_kmeans_max_k': 5,
        'decon_kmeans_proximity_threshold': 2,
        'decon_kmeans_min_cluster_detections': 5,
        # steps
        'apply_drift_correction': 0,
        'save_binned_detections': 0,
        'mask_on_density': 0,
        'make_quality_metrics': 0,
        'plot_per_fiducial_fitting': 0,
        'plot_fiducial_correlations': 0,
        'plot_summary_stats': 0,
        'plot_detections': 0,
        'plot_fiducials': 0,
        'zstep_correct_fiducials': 0,
        'deconvolve_z': 0,
        'drift_correct_detections': 0
    }
    return config

def config_update(config: dict, new_config: dict) -> dict:
    for key in new_config:
        if key in config:
            config[key] = new_config[key]
        else:
            raise KeyError(f"Unknown entry: {key} found in config file")
    return config

def config_validate(config: dict) -> int:
    ret = 1
    # Check existence of detections file
    if not os.path.exists(config['detections_file']):
        logging.error(f"Detections file {config[config['detections_file']]} not found")
        ret = 0
    # Check that output directory exists
    if not os.path.exists(config['output_dir']):
        logging.error(f"Output directory {config['output_dir']} not found")
        ret = 0
    return ret

def config_validate_detections(df: pd.DataFrame, config: dict) -> int:
    ret = 1
    # Check existence of columns
    required_cols = [config['frame_col'],
                     config['image_id_col'],
                     config['z_step_col'],
                     config['cycle_col'],
                     config['time_point_col'],
                     config['x_col'],
                     config['y_col'],
                     config['z_col'],
                     config['x_sd_col'],
                     config['y_sd_col'],
                     config['z_sd_col'],
                     config['photons_col'],
                     config['chisq_col']]
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Column {col} not found in detections file")
            ret = 0

    # Check that ranges are adhered to
    for quantity in ['frame',
                  'z_step',
                  'cycle',
                  'time_point']:
        range_key = f"{quantity}_range"
        quantity_col = f"{quantity}_col"
        min_range = int(config[range_key].split('-')[0])
        max_range = int(config[range_key].split('-')[1])
        min_quantity = df[config[quantity_col]].min()
        max_quantity = df[config[quantity_col]].max()
        if min_quantity < min_range or max_quantity > max_range:
            logging.error(f"Range for {quantity} out of bounds: Either min({quantity})={min_quantity} < {min_range} or max({quantity})={max_quantity} > {max_range}")
            ret = 0
        if min_quantity != min_range or max_quantity != max_range:
            logging.warning(f"Range for {quantity} not filled: Either min({quantity})={min_quantity} != {min_range} or max({quantity})={max_quantity} != {max_range}")
    # Check max for image-ID
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
    max_image_id = df[config['image_id_col']].max()
    if max_image_id > total_frames:
        logging.error(f"Max {config['image_id_col']} = {max_image_id} exceeds total possible frames {total_frames}")
        ret = 0
    # Check that image_id_col is correct with respect to frame, z_step, cycle, time_point
    expected_image_id = (df[config['frame_col']] +
                        (df[config['z_step_col']] - min_z_step) * num_frames +
                        (df[config['cycle_col']] - min_cycle) * frames_per_cycle +
                        (df[config['time_point_col']] - min_time_point) * frames_per_cycle * num_cycles)
    if not expected_image_id.equals(df[config['image_id_col']]):
        logging.error(f"Image-ID column does not match frame, z-step, cycle, time-point columns")
        idx = expected_image_id != df[config['image_id_col']]
        logging.error(f"Expected: {expected_image_id[idx].to_numpy()}")
        logging.error(f"Actual: {df[config['image_id_col']][idx].to_numpy()}")
        ret = 0
    return ret

def config_print(config: dict) -> None:
    for key in config:
        logging.info(f"{key}: {config[key]}")