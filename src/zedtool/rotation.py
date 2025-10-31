#!/usr/bin/env python3

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mutual_info_score

def rotation_correct_detections(df: pd.DataFrame, df_fiducials: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info('rotation_correct_detections')
    MIN_FIDUCIAL_DETECTIONS = 100
    min_time_point, max_time_point = map(int, config['time_point_range'].split('-'))
    timepoints = df[config['time_point_col']].values
    fiducial_label = df['label'].values
    fiducial_labels = df_fiducials['label'].values
    n_labels = len(fiducial_labels)
    is_fiducial = df['is_fiducial'].values
    xyz = np.column_stack((df[config['x_col']].values, df[config['y_col']].values, df[config['z_col']].values))
    xyz_1 = np.zeros((n_labels, 3))
    xyz_2 = np.zeros((n_labels, 3))
    # Loop over all steps and find and apply translation and rotation correction at each time-point boundary
    # Tries to align successive time-points with those that came before.
    for timepoint in range(min_time_point+1, max_time_point+1):
        logging.info(f'Correcting rotation at time point {timepoint}')
        idx_1 = (timepoints < timepoint) & is_fiducial
        idx_2 = (timepoints == timepoint) & is_fiducial
        if np.sum(idx_1) < MIN_FIDUCIAL_DETECTIONS or np.sum(idx_2) < MIN_FIDUCIAL_DETECTIONS:
            logging.warning(f'Insufficient detections on one side of time point {timepoint}')
            logging.warning(f'{np.sum(idx_1)} detections before, {np.sum(idx_2)} detections at time point')
            continue
        is_valid_fiducial = np.zeros(n_labels, dtype=bool)
        for j in range(n_labels):
            fiducial_idx = (fiducial_label == fiducial_labels[j])
            idx_1j = idx_1 & fiducial_idx
            idx_2j = idx_2 & fiducial_idx
            if np.sum(idx_1j) >= MIN_FIDUCIAL_DETECTIONS and np.sum(idx_2j) >= MIN_FIDUCIAL_DETECTIONS:
                if config['verbose']:
                    logging.info(f'Using fiducial {fiducial_labels[j]} for rotation correction at time point {timepoint}. ndetections before: {np.sum(idx_1j)}, at time point: {np.sum(idx_2j)}')
                is_valid_fiducial[j] = True
                xyz_1[j,:] = np.nanmean(xyz[idx_1j,:], axis=0)
                xyz_2[j,:] = np.nanmean(xyz[idx_2j,:], axis=0)
            else:
                logging.info(f'Skipping fiducial {fiducial_labels[j]} for rotation correction at time point {timepoint}. ndetections before: {np.sum(idx_1j)}, at time point: {np.sum(idx_2j)}')
        if np.sum(is_valid_fiducial) < 3:
            logging.error(f'Insufficient valid fiducials for rotation correction at time point {timepoint}')
            logging.error(f'{np.sum(is_valid_fiducial)} valid fiducials found')
            continue
        xyz_1_valid = xyz_1[is_valid_fiducial,:]
        xyz_2_valid = xyz_2[is_valid_fiducial,:]
        rotation_matrix, translation, x_aligned, rmse = euclidean_rigid_alignment_3d(xyz_2_valid, xyz_1_valid)
        # check translation and rotation for nans or infs
        if np.any(np.isnan(translation)) or np.any(np.isinf(translation)):
            logging.error(f'Invalid translation vector at time point {timepoint}')
            continue
        if np.any(np.isnan(rotation_matrix)) or np.any(np.isinf(rotation_matrix)):
            logging.error(f'Invalid rotation matrix at time point {timepoint}')
            continue
        # Convert rotation matrix to axis-angle representation for logging
        axis, angle = rotation_axis_angle(rotation_matrix)
        # Apply the rotation and translation to all points in df at timepoint
        idx = (timepoints == timepoint)
        xyz_rotated = (rotation_matrix @ xyz[idx,:].T).T + translation
        xyz[idx,:] = xyz_rotated
        logging.info(f'Applied rotation and translation at time point {timepoint}: RMSE = {rmse:.3f} nm')
        logging.info(f'Translation vector at time point {timepoint}: [x,y,z] = {translation} nm')
        logging.info(f'Rotation matrix at time point {timepoint}: theta = {angle*180/np.pi:.4f} degrees around axis [x,y,z] = {axis}')
        if config['verbose']:
            logging.info(f'Rotation matrix:\n{rotation_matrix}')
    # end for timepoint
    # Update the dataframe with rotated coordinates
    df[config['x_col']] = xyz[:,0]
    df[config['y_col']] = xyz[:,1]
    df[config['z_col']] = xyz[:,2]
    return df

def euclidean_rigid_alignment_3d(X, Y):
    """
    Compute the rigid Euclidean (rotation + translation) transform
    that best aligns X onto Y.

    Parameters
    ----------
    X : ndarray of shape (n_points, 3)
        Source points
    Y : ndarray of shape (n_points, 3)
        Target points

    Returns
    -------
    R : ndarray of shape (3,3)
        Rotation matrix
    t : ndarray of shape (3,)
        Translation vector
    X_aligned : ndarray of shape (n_points, 3)
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

def rotation_axis_angle(R):
    """
    Convert a 3×3 rotation matrix into axis-angle representation.

    Parameters
    ----------
    R : ndarray (3,3)
        Rotation matrix

    Returns
    -------
    axis : ndarray (3,)
        Unit vector along the axis of rotation
    angle : float
        Rotation angle in radians
    """
    # Ensure R is a valid rotation
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), "R is not orthogonal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "R is not a proper rotation"

    angle = np.arccos((np.trace(R) - 1) / 2)

    if np.isclose(angle, 0):
        # No rotation
        return np.array([1, 0, 0]), 0.0
    elif np.isclose(angle, np.pi):
        # Special case: 180° rotation → axis ambiguous
        axis = np.sqrt((np.diag(R) + 1) / 2.0)
        return axis / np.linalg.norm(axis), angle
    else:
        axis = np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ]) / (2*np.sin(angle))
        return axis / np.linalg.norm(axis), angle

def make_rotation_quality_metric(df_fiducials: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info('make_rotation_quality_metric')
    # Ensure there are no NaNs in the relevant columns
    idx = (~df_fiducials['x_mean'].isna()) & (~df_fiducials['y_mean'].isna()) & \
          (~df_fiducials['x_sd'].isna()) & (~df_fiducials['y_sd'].isna())

    if np.sum(idx) < 2:
        logging.warning('Insufficient fiducials with valid data for rotation quality metric')
        return None

    # use mutual information and permutation test to see if x_sd, y_sd and z_sd are functions of x_mean and y_mean
    y = df_fiducials['y_sd'].values
    x = df_fiducials['x_mean'].values
    pvalue_yx,stat_yx = dependence_test(x, y)
    logging.info(f"stat_yx: norm_mutual_information = {stat_yx}, p_value = {pvalue_yx}")

    y = df_fiducials['x_sd'].values
    x = df_fiducials['y_mean'].values
    pvalue_xy,stat_xy = dependence_test(x, y)
    logging.info(f"stat_xy: norm_mutual_information = {stat_xy}, p_value = {pvalue_xy}")

    y = df_fiducials['z_sd'].values
    x = df_fiducials['x_mean'].values
    pvalue_zx,stat_zx = dependence_test(x, y)
    logging.info(f"stat_zx: norm_mutual_information = {stat_zx}, p_value = {pvalue_zx}")

    y = df_fiducials['z_sd'].values
    x = df_fiducials['y_mean'].values
    pvalue_zy,stat_zy = dependence_test(x, y)
    logging.info(f"stat_zy: norm_mutual_information = {stat_zy}, p_value = {pvalue_zy}")

    results = {
        'pvalue_yx': pvalue_yx,
        'stat_yx': stat_yx,
        'pvalue_zx': pvalue_zx,
        'stat_zx': stat_zx,
        'pvalue_xy': pvalue_xy,
        'stat_xy': stat_xy,
        'pvalue_zy': pvalue_zy,
        'stat_zy': stat_zy
    }
    df = pd.DataFrame([results])
    return df

def dependence_test(x, y):
    """
    Test for dependence between x and y using mutual information and a permutation test.
    Args:
        x: independent variable
        y: possibly dependent variable

    Returns:
        p_value: p-value from permutation test
        normalised_mi: normalised mutual information score
    """
    n_perm = 10000
    perm_mi = np.zeros(n_perm)
    x_bins = np.digitize(x, bins=np.histogram_bin_edges(x, bins='auto'))
    y_bins = np.digitize(y, bins=np.histogram_bin_edges(y, bins='auto'))

    # Calculate the mutual information score
    mi_score = mutual_info_score(x_bins, y_bins)

    # Normalise the MI score with the entropy of y, H(y).
    h_y = -np.sum(np.bincount(y_bins) / len(y_bins) * np.log(np.bincount(y_bins) / len(y_bins) + 1e-9))
    normalised_mi = mi_score / h_y if h_y != 0 else 0.0

    for i in range(n_perm):
        y_perm = np.random.permutation(y_bins)
        perm_mi[i] = mutual_info_score(x_bins, y_perm)

    p_value = np.mean(perm_mi >= mi_score)
    return p_value, normalised_mi

