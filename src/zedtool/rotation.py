#!/usr/bin/env python3

import numpy as np
import pandas as pd
import logging
from typing import Tuple
import sklearn
import os
import scipy
import multiprocessing
import matplotlib.pyplot as plt


def rotation_correct_detections(df: pd.DataFrame, df_fiducials: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info('rotation_correct_detections')
    MIN_FIDUCIAL_DETECTIONS = 100
    min_time_point, max_time_point = map(int, config['time_point_range'].split('-'))
    n_fiducials = len(df_fiducials)
    timepoints = df[config['time_point_col']].values
    fiducial_label = df['label'].values
    is_fiducial = df['is_fiducial'].values
    x = df[config['x_col']].values
    y = df[config['y_col']].values
    xy_1 = np.zeros((n_fiducials, 2))
    xy_2 = np.zeros((n_fiducials, 2))
    # Loop over all steps and find and apply translation and rotation correction at each step boundary
    for timepoint in range(min_time_point+1, max_time_point+1):
        logging.info(f'Correcting rotation at time point {timepoint}')
        idx_1 = (timepoints < timepoint) & (is_fiducial)
        idx_2 = (timepoints == timepoint) & (is_fiducial)
        if np.sum(idx_1) < MIN_FIDUCIAL_DETECTIONS or np.sum(idx_2) < MIN_FIDUCIAL_DETECTIONS:
            logging.warning(f'insufficient detections on one side of time point: {timepoint}')
            logging.warning(f'{np.sum(idx_1)} detections before, {np.sum(idx_2)} detections at time point')
            continue
        is_valid_fiducial = np.zeros(n_fiducials, dtype=bool)
        for j in range(n_fiducials):
            fiducial_idx = (fiducial_label == (j+1))
            idx_1j = idx_1 & fiducial_idx
            idx_2j = idx_2 & fiducial_idx
            if np.sum(idx_1j) >= MIN_FIDUCIAL_DETECTIONS and np.sum(idx_2j) >= MIN_FIDUCIAL_DETECTIONS:
                # logging.info(f'Using fiducial {j+1} for rotation correction at time point {timepoint}. ndetections before: {np.sum(idx_1j)}, at time point: {np.sum(idx_2j)}')
                is_valid_fiducial[j] = True
                xy_1[j,0] = np.nanmean(x[idx_1j])
                xy_1[j,1] = np.nanmean(y[idx_1j])
                xy_2[j,0] = np.nanmean(x[idx_2j])
                xy_2[j,1] = np.nanmean(y[idx_2j])
        xy_1_valid = xy_1[is_valid_fiducial,:]
        xy_2_valid = xy_2[is_valid_fiducial,:]
        R, t, X_aligned, rmse = euclidean_rigid_alignment(xy_2_valid, xy_1_valid)
        # Apply the rotation and translation to all points in df at timepoint
        idx = df[config['time_point_col']] == timepoint
        xy = np.column_stack((x[idx], y[idx]))
        xy_rotated = (R @ xy.T).T + t
        df.loc[idx, config['x_col']] = xy_rotated[:,0]
        df.loc[idx, config['y_col']] = xy_rotated[:,1]
        logging.info(f'Applied rotation and translation at time point {timepoint}: RMSE = {rmse:.3f} nm')
        logging.info(f'Rotation matrix at time point {timepoint}: [cos(theta), sin(theta)] = {R[0,:]}')
        logging.info(f'Translation vector at time point {timepoint}: [x, y] = {t}')
    return df

def euclidean_rigid_alignment(X, Y):
    """
    Compute the rigid Euclidean (rotation + translation) transform
    that best aligns X onto Y.

    Parameters
    ----------
    X : ndarray of shape (n_points, 2)
        Source points
    Y : ndarray of shape (n_points, 2)
        Target points

    Returns
    -------
    R : ndarray of shape (2,2)
        Rotation matrix
    t : ndarray of shape (2,)
        Translation vector
    X_aligned : ndarray of shape (n_points, 2)
    X after applying rotation and translation
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Center the points (subtract centroids)
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    X0 = X - mu_X
    Y0 = Y - mu_Y

    # Compute covariance matrix
    H = X0.T @ Y0

    # SVD of covariance
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = mu_Y - R @ mu_X

    # Apply transformation
    X_aligned = (R @ X.T).T + t

    # Compute quality metric (RMSE)
    distances = np.sqrt(np.sum((X_aligned - Y) ** 2, axis=1))
    rmse = np.sqrt(np.mean(distances ** 2))

    return R, t, X_aligned, rmse
