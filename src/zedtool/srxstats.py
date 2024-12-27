#!/usr/bin/env python3
from pickle import FALSE
import pandas as pd
import numpy as np
from typing import Tuple

def apply_correction(df: pd.DataFrame, tdz: np.ndarray) -> pd.DataFrame:
    # Apply z-correction to detections
    dz = np.zeros((int(np.max(tdz[:, 0]) + 1)))
    dz[tdz[:,0].astype(int)] = tdz[:,1]
    df['z'] = df['z'] + dz[df['image-ID'].values]
    return df

def remove_corrections(df: pd.DataFrame, tdz: np.ndarray) -> pd.DataFrame:
    # Remove z-correction to detections
    dz = np.zeros((int(np.max(tdz[:, 0]) + 1)))
    dz[tdz[:,0].astype(int)] = tdz[:,1]
    df['z'] = df['z'] - dz[df['image-ID'].values]
    return df

def find_cycle_boundaries(df: pd.DataFrame) -> np.ndarray:
    # Find the boundaries of cycles in the dataframe
    cycle_boundaries = np.where(np.diff(df_masked['cycle']))[0] + 1
    cycle_boundaries = np.insert(cycle_boundaries, 0, 0)
    cycle_boundaries = np.append(cycle_boundaries, df_masked.shape[0]-1)
    return cycle_boundaries

def tdz_by_timepoints(df: pd.DataFrame, det_xyz: np.ndarray, timepoint_intervals: np.ndarray) -> np.ndarray:
    # Returns a 2D array with columns t, dz between successive time-points and dz by time-point from start
    print("Computing tdz by cycles...")
    tdz = np.zeros((timepoint_intervals.shape[0]-1,3))
    z_mean, t = z_means_by_marker(det_xyz, df['time-point'].values)
    t0 = df['image-ID'].values[timepoint_intervals[:, 0]]
    t1 = df['image-ID'].values[timepoint_intervals[:, 1]]
    t = (t0 + t1) / 2
    tdz[:,0] = (t[:-1] + t[1:]) / 2
    tdz[:,1] = z_mean[1:] - z_mean[:-1]
    tdz[:,2] = z_mean[1:] - z_mean[0]
    tdz0 = np.array([0, 0, 0])
    tdz = np.vstack((tdz0,tdz))
    return tdz

def tdz_from_intervals(tdz_time_points: np.ndarray, time_point_intervals: np.ndarray, df: np.ndarray) -> np.ndarray:
    # Returns a dataframe with columns t, dz by time-point from start
    print("Computing df_tdz from intervals...")
    total_time_points = np.sum(time_point_intervals[:,1]-time_point_intervals[:,0])
    # df_tdz = pd.DataFrame(columns=['t', 'dz'], index = np.arange(total_time_points))
    tdz = np.zeros((total_time_points,2))
    offset=0
    for j in range(time_point_intervals.shape[0]):
        t0 = time_point_intervals[j, 0]
        t1 = time_point_intervals[j, 1]
        tdz[offset:offset+t1-t0,0] = np.arange(t0, t1)
        tdz[offset:offset+t1-t0,1] = tdz_time_points[j,2]
        offset = offset + t1-t0
    tdz[:,0] = df['image-ID'].values[tdz[:,0].astype(int)]
    t, indices = np.unique(tdz[:,0], return_index=True)
    tdz = tdz[indices,:]
    return tdz

def extract_z_correction(df_cor: pd.DataFrame, df_orig: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Returns z-corrections from two sets of detections
    # First it finds the common rows between the two dataframes and then extracts the z-corrections
    print("Extracting SRX z-corrections...")
    # nrows = df_cor['image-ID'].max()
    # Find the rows from df_cor in df_orig using a unqique ID made from image-ID*1e6+photon-count
    # This is because photon-count is unique within a time-point
    # The unique ID is used to index the rows
    unique_id_cor = df_cor['image-ID']*1e6+df_cor['photon-count']
    df_cor['row_id'] = unique_id_cor
    unique_id_orig = df_orig['image-ID']*1e6+df_orig['photon-count']
    df_orig['row_id'] = unique_id_orig
    # make row_id an index
    df_orig.set_index('row_id', inplace=True)
    df_cor.set_index('row_id', inplace=True)
    # Find the intersection of the unique_ids
    common_ids = np.intersect1d(unique_id_cor, unique_id_orig)
    df_orig = df_orig.loc[common_ids]
    df_cor = df_cor.loc[common_ids]
    # remove indexes
    df_orig.reset_index(inplace=True)
    df_cor.reset_index(inplace=True)
    # Extract dz from matching rows
    tdz = np.zeros((df_orig.shape[0], 2))
    tdz[:,1] = df_cor['z'] - df_orig['z']
    tdz[:,0] = df_orig['image-ID']
    t, indices = np.unique(tdz[:,0], return_index=True)
    tdz = tdz[indices,:]
    df_tdz = pd.DataFrame(tdz, columns=['t', 'dz'])
    df_cor['z_orig'] = df_orig['z']
    return df_tdz,df_cor

def tdz_by_cycles(df: pd.DataFrame, det_xyz: np.ndarray, cycle_intervals: np.ndarray) -> np.ndarray:
    # Returns a 2D array with columns t, dz between successive cycles and dz by cycle from start
    print("Computing tdz by time points and cycles")
    # Make cycle_zstep
    max_cycle = df['cycle'].max()
    # Round to highest power of 10
    timestep_mult = int(10 ** (np.ceil(np.log10(max_cycle))))
    timestep_cycle = df['time-point'] * timestep_mult + df['cycle']
    z_mean, t = z_means_by_marker(det_xyz, timestep_cycle)
    t0 = df['image-ID'].values[cycle_intervals[:, 0]]
    t1 = df['image-ID'].values[cycle_intervals[:, 1]]
    t = (t0 + t1) / 2
    tdz = np.zeros((t.shape[0]-1,3))
    if t.shape[0]!=z_mean.shape[0]:
        print("Warning: t.shape[0]!=tdz.shape[0]")
        print(f"t.shape[0]: {t.shape[0]}, z_mean.shape[0]: {z_mean.shape[0]}")
    tdz[:,0] = (t[:-1] + t[1:]) / 2
    tdz[:,1] = (z_mean[1:] - z_mean[:-1])[0:tdz.shape[0]]
    tdz[:,2] = (z_mean[1:] - z_mean[0])[0:tdz.shape[0]]
    tdz0 = np.array([0, 0, 0])
    tdz = np.vstack((tdz0,tdz))
    return tdz

def tdz_by_zsteps(df: pd.DataFrame, det_xyz: np.ndarray, zstep_intervals: np.ndarray) -> np.ndarray:
    # Returns a 2D array with columns t, dz between successive zsteps and dz by zstep from start
    # Broken when there is more than one time-point
    print("Computing tdz by cycles and z-steps...")
    t0 = df['image-ID'].values[zstep_intervals[:, 0]]
    t1 = df['image-ID'].values[zstep_intervals[:, 1]]
    mid_points = (t0 + t1) / 2
    # Make cycle_zstep
    max_zstep = df['z-step'].max()
    # Round to highest power of 10
    cycle_mult = int(10 ** (np.ceil(np.log10(max_zstep))))
    cycle_zstep = df['cycle'] * cycle_mult + df['z-step']

    z_mean, t = z_means_by_marker(det_xyz, cycle_zstep)
    n_cycles = 1 + df['cycle'].max()-df['cycle'].min()
    n_zsteps = 1 + df['z-step'].max()-df['z-step'].min()
    tdz = np.zeros(((n_cycles-1) * n_zsteps,2))
    if n_zsteps*n_cycles > zstep_intervals.shape[0]:
        print("Error: n_zsteps*n_cycles > zstep_intervals.shape[0]")
        print(f"n_zsteps: {n_zsteps}, n_cycles: {n_cycles}, zstep_intervals.shape[0]: {zstep_intervals.shape[0]}")
        exit(1)
    for i in range(n_cycles - 1):
        for j in range(n_zsteps):
            idx_start = i * n_zsteps + j
            idx_end = (i+1) * n_zsteps + j
            tdz[idx_start,0] = (mid_points[idx_end] + mid_points[idx_start])/2
            tdz[idx_start,1] = z_mean[idx_end] - z_mean[idx_start]
    tdz0 = np.array([0, 0])
    tdz = np.vstack((tdz0,tdz))
    return tdz

def make_intervals(boundaries: np.ndarray) -> np.ndarray:
    new_intervals = np.column_stack((boundaries[:-1], boundaries[1:]))
    return new_intervals

def z_means_by_marker(det_xyz: np.ndarray, marker: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Finds the mean z for each unique marker value
    # Returns a 2D array with columns z_mean, marker value
    # Find unique values and the corresponding indices
    t, indices = np.unique(marker, return_inverse=True)
    # Use np.bincount to accumulate the values and count for each unique index
    sum = np.bincount(indices, weights=det_xyz[:, 2])
    count = np.bincount(indices)
    # Avoid division by zero and calculate mean values
    z_mean = np.divide(sum, count, out=np.zeros_like(sum), where=count != 0)
    return z_mean,t

