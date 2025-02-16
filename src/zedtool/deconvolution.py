#!/usr/bin/env python3

import numpy as np
import pandas as pd
import logging
import sklearn
import os
import matplotlib.pyplot as plt

def deconvolve_z_within_time_point(df: pd.DataFrame, df_fiducials: pd.DataFrame, n_xy: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray, config: dict) -> pd.DataFrame:
    logging.info("deconvolve_z_within_time_point")
    min_time_point, max_time_point = map(int, config['time_point_range'].split('-'))
    for time_point in range(min_time_point, max_time_point + 1):
        logging.info(f"Deconvolving z for time-point {time_point}")
        idx = df[config['time_point_col']] == time_point
        df_time_point = df.loc[idx, :]
        df_time_point = deconvolve_z(df_time_point, df_fiducials, n_xy, x_idx, y_idx, config)
        df.loc[idx, :] = df_time_point
    return df

def deconvolve_z(df: pd.DataFrame, df_fiducials: pd.DataFrame, n_xy: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray, config: dict) -> pd.DataFrame:
    logging.info("deconvolve_z")
    bin_threshold = 10
    # TODO: Optionally specify sigma_fit and sigma_target
    sigma_fit = np.median(df_fiducials['z_sd'])
    sigma_target = (np.median(df_fiducials['x_sd']) + np.median(df_fiducials['y_sd'])) / 2
    logging.info(f"Squeezing {np.sum(n_xy > bin_threshold)} bins. sigma_fit = {sigma_fit}\tsigma_target = {sigma_target}")
    # Find all the bins in n_xy that have more than bin_threshold detections and loop over them with their x,y coords
    for x in range(n_xy.shape[0]):
        for y in range(n_xy.shape[1]):
            if n_xy[x, y] > bin_threshold:
                # Find all detections in this bin
                idx = (x_idx == x) & (y_idx == y)
                # Get the z values of these detections
                z = df.loc[idx, 'z'].to_numpy()
                # Get the z values of the fiducials in this bin
                # Deconvolve the z values
                # z_deconv = deconvolve_mog(z, sigma_fit, sigma_target, config)
                z_deconv = deconvolve_kmeans(z, sigma_fit, sigma_target, config)
                # Replace the z values in the dataframe
                df.loc[idx, 'z'] = z_deconv
    return df

def deconvolve_kmeans(z: np.ndarray, sigma_fit: float, sigma_target: float, config: dict) -> np.ndarray:
    logging.info("deconvolve_kmeans")
    # Deconvolve z using the x,y distribution from the fiducials as a target distribution
    # Squeezes the z distribution of detections in each bin to the x,y distribution of fiducials
    # Uses k-means to find the peaks in the z distribution
    # Increase k until the peaks are not too small and they are not too close together
    max_k = 4
    proximity_threshold = 1.5
    k = 0
    for i in range(1, max_k + 1):
        kmeans = sklearn.cluster.KMeans(n_clusters=i, random_state=0, n_init=1)
        kmeans.fit(z.reshape(-1, 1))
        cluster_means = kmeans.cluster_centers_.flatten()
        cluster_sds = np.zeros(i)
        # TODO: exclude tiny and narrow cluster from squeezing but not from k limiting
        # compute the standard deviation of each cluster
        for j in range(i):
            cluster_sds[j] = np.std(z[kmeans.labels_ == j])
        # check if any cluster has a standard deviation less than the target
        if np.any(cluster_sds < sigma_target):
            break
        # check if any two clusters are within proximity_threshold * their SDs of each other
        if np.any(np.abs(cluster_means[1:] - cluster_means[:-1]) < proximity_threshold * (cluster_sds[1:] + cluster_sds[:-1])):
            break
        k = i
        kmeans_best = kmeans

    logging.info(f"Points: {len(z)} Clusters: {k}")
    if k == 0:
        return z
    # Squeeze each z value to the cluster with the closest mean
    z_deconv = np.zeros_like(z)
    for i, mean in enumerate(kmeans_best.cluster_centers_.flatten()):
        mask = (kmeans_best.labels_ == i)
        z_deconv[mask] = mean + (z[mask] - mean) * (sigma_target / sigma_fit)
    return z_deconv

def deconvolve_mog(z: np.ndarray, sigma_fit: float, sigma_target: float, config: dict) -> np.ndarray:
    logging.info("deconvolve_mog")
    max_trial_models = 10
    max_models = 4
    # Fit GMM with different numbers of components and select the best model based on BIC
    best_gmm = None
    best_bic = np.inf
    for n_components in range(1, max_trial_models + 1):
        gmm = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=0)
        gmm.precisions_init = np.full(n_components, 1 / sigma_fit ** 2)
        gmm.fit(z.reshape(-1, 1))
        bic = gmm.bic(z.reshape(-1, 1))
        print(f"n_components = {n_components}\tBIC = {bic}")
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    if best_gmm.n_components > max_models:
        logging.warning(f"Best GMM has {best_gmm.n_components} components, which is more than the maximum of {max_models}")
        return z
    # Adjust z to have the same means but with sigma_target rather than sigma_fit
    means = best_gmm.means_.flatten()
    weights = best_gmm.weights_
    labels = best_gmm.predict(z.reshape(-1, 1))

    z_adjusted = np.zeros_like(z)
    for i, mean in enumerate(means):
        mask = (labels == i)
        z_adjusted[mask] = mean + (z[mask] - mean) * (sigma_target / sigma_fit)
    # plot a histogram of the z values before and after deconvolution
    plot_deconvolution(z, z_adjusted, config)
    return z_adjusted

def plot_deconvolution(z: np.ndarray, z_adjusted: np.ndarray, config: dict):
    logging.info("plot_deconvolution")
    plt.hist(z, bins=100, alpha=0.5, label='Before')
    plt.hist(z_adjusted, bins=100, alpha=0.5, label='After')
    plt.legend()
    plt.savefig(os.path.join(config['output_dir'], 'deconvolution.png'))
    plt.close()