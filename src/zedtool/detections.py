#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Tuple
import logging

def make_density_mask_2d(n_xy: np.ndarray, config: dict) -> np.ndarray:
    # Make a mask for the density of detections in the x-y plane
    n_min = 10
    n_max = 10**np.quantile(np.log10(n_xy[n_xy >= n_min]), 0.95)
    if 'n_max_cutoff' in config:
        n_max = config['n_max_cutoff']
    if 'n_min_cutoff' in config:
        n_min = config['n_min_cutoff']
    mask_xy = np.zeros(n_xy.shape)
    mask_xy[ (n_xy >= n_min) & (n_xy <= n_max) ] = 1
    return mask_xy

def make_image_index(det_xyz, x_bins, y_bins, z_bins) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_idx = np.digitize(det_xyz[:,0], x_bins)-1
    y_idx = np.digitize(det_xyz[:,1], y_bins)-1
    z_idx = np.digitize(det_xyz[:,2], z_bins)-1
    return x_idx, y_idx, z_idx

def im_to_detection_entry(im: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    # Mask detections in the x-y plane
    return(im[x_idx,y_idx])

def mask_detections(mask_xy: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    # Mask detections in the x-y plane
    idx = mask_xy[x_idx,y_idx]
    return(idx.astype(bool))

def bin_detections(det_xyz: np.array,resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logging.info("Binning detections...")
    # bin detections into a 3D grid
    x = det_xyz[:,0]
    y = det_xyz[:,1]
    z = det_xyz[:,2]
    # Add an extra bin to the end for histogramdd() then remove it later
    x_bins = np.arange(x.min(), x.max() + resolution, resolution)
    y_bins = np.arange(y.min(), y.max() + resolution, resolution)
    z_bins = np.arange(z.min(), z.max() + resolution, resolution)
    counts_xyz = np.histogramdd([x, y, z], bins=[x_bins, y_bins, z_bins])
    return counts_xyz[0], x_bins[:-1], y_bins[:-1], z_bins[:-1]

def bins3d_to_stats2d(counts_xyz: np.ndarray, z_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Calculate moments and sd from 3D bins
    n_xy = np.sum(counts_xyz,axis=2)
    moment_1_xy = np.sum(counts_xyz * z_bins, axis=2) / (n_xy + 1)
    moment_2_xy = np.sum(counts_xyz * (z_bins**2), axis=2) / (n_xy + 1)
    var_xy = moment_2_xy - moment_1_xy ** 2
    sd_xy = np.sqrt(var_xy)
    return n_xy, moment_1_xy, sd_xy

def median_by_time(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Group by time-point and calculate the median for x, y, and z
    grouped = df.groupby(config['image_id_col']).median()
    # Extract the median values for x, y, and z
    median_values = grouped[[config['x_col'], config['y_col'], config['z_col']]].values
    ret = np.full((3,np.max(grouped.index)+1),np.inf)
    ret[:,grouped.index] = np.transpose(median_values)
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
    # Split colnames and ranges into lists
    columns = colnames.split(',')
    range_list = ranges.split(',')

    # Ensure columns and ranges have the same length
    if len(columns) != len(range_list):
        raise ValueError("The number of columns and ranges must match")

    # Start with the full DataFrame
    filtered_df = df.copy()
    logging.info(f"Initial number of rows: {filtered_df.shape[0]}")
    # TODO: Build up idx and filter once.

    # Apply filters for each column and range
    for col, r in zip(columns, range_list):
        # Parse the range (e.g., "1.0-3.2" -> 1.0, 3.2)
        low, high = map(float, r.split('-'))
        # Filter the DataFrame
        filtered_df = filtered_df[(filtered_df[col] >= low) & (filtered_df[col] <= high)]
        logging.info(f"Filtered {col} between {low} and {high}: {filtered_df.shape[0]} rows")
    return filtered_df

def reorder_sweeps(df: pd.DataFrame, config: dict) -> pd.DataFrame: # TODO:
    # Reorder sweeps based on the z-step
    return df
