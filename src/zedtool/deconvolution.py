#!/usr/bin/env python3

import numpy as np
import pandas as pd
import logging
import sklearn
import os
import scipy
import multiprocessing
import matplotlib.pyplot as plt
# from threadpoolctl import threadpool_limits

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
    if config['debug']:
        outdir = os.path.join(config['output_dir'], 'cluster_plots')
        os.makedirs(outdir, exist_ok=True)
    # multiprocessing_works_with_sklearn = False
    # if multiprocessing_works_with_sklearn and config['multiprocessing']:
    if config['multiprocessing']:
        return deconvolve_z_parallel(df, df_fiducials, n_xy, x_idx, y_idx, config)
    logging.info("deconvolve_z")
    bin_threshold = config['decon_bin_threshold']
    min_cluster_sd = config['decon_min_cluster_sd']
    sd_shrink_ratio = config['decon_sd_shrink_ratio']
    logging.info(f"Squeezing {np.sum(n_xy > bin_threshold)} bins.")
    # Find all the bins in n_xy that have more than bin_threshold detections and loop over them with their x,y coords
    for x in range(n_xy.shape[0]):
        for y in range(n_xy.shape[1]):
            if n_xy[x, y] > bin_threshold:
                # Find all detections in this bin
                idx = (x_idx == x) & (y_idx == y)
                # Get the z values of these detections
                z = df.loc[idx, 'z'].to_numpy()
                # Deconvolve the z values
                # z_deconv = deconvolve_mog(z, sigma_fit, sigma_target, config)
                z_deconv = deconvolve_kmeans(z, min_cluster_sd, sd_shrink_ratio, config)
                # Replace the z values in the dataframe
                df.loc[idx, 'z'] = z_deconv
    return df

def deconvolve_z_parallel(df: pd.DataFrame, df_fiducials: pd.DataFrame, n_xy: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray, config: dict) -> pd.DataFrame:
    logging.info("deconvolve_z_parallel")
    min_cluster_sd = config['decon_min_cluster_sd']
    sd_shrink_ratio = config['decon_sd_shrink_ratio']
    nbins = np.sum(n_xy > config['decon_bin_threshold'])
    if config['debug']:
        decon_summary_stats = os.path.join(config['output_dir'], f'cluster_plots/summary.tsv')
        with open(decon_summary_stats, 'w') as f:
            f.write('npeaks\tsize\tmean\tsd\tmin\tmax\tmedian\n')

    # bin_x and bin_y are coordinates of bins with more than bin_threshold detections
    bin_x, bin_y = np.where(n_xy > config['decon_bin_threshold'])

    tasks = [(df.loc[(x_idx == bin_x[i]) & (y_idx == bin_y[i]), 'z'].to_numpy(),
              min_cluster_sd, sd_shrink_ratio, config) for i in range(nbins)]

    with multiprocessing.Pool(int(config['num_threads'])) as pool:
        results = pool.starmap(deconvolve_kmeans, tasks)

    for i,z_deconv in zip(np.arange(nbins),results):
        df.loc[(x_idx == bin_x[i]) & (y_idx == bin_y[i]), 'z'] = z_deconv

    if config['debug']:
        # scatter plot of decon_summary_stats showing sd versus size, coloured by npeaks
        df_summary = pd.read_csv(decon_summary_stats, sep='\t')
        plt.figure()
        plt.scatter(df_summary['size'], df_summary['sd'], c=df_summary['npeaks'], cmap='viridis')
        plt.xlabel('Size')
        plt.ylabel('Stddev')
        plt.title('Stddev vs Size')
        plt.colorbar(label='Number of Peaks')
        outfile = os.path.join(config['output_dir'], 'cluster_plots', f'summary.{config['plot_format']}')
        plt.savefig(outfile)
        plt.close()

    return df

def deconvolve_kmeans(z: np.ndarray, min_cluster_sd: float, sd_shrink_ratio: float, config: dict) -> np.ndarray:
    # logging.info("deconvolve_kmeans_sklearn")
    # Squeezes the z distribution of the peaks in z by an amount sd_shrink_ratio.
    # Only shrink if the sd of the cluster is bigger than min_cluster_sd to begin with
    # Uses k-means to find the peaks in the z distribution
    # Increase k until the peaks are too small or close together. Then, if possible, squeeze
    max_k = config['decon_kmeans_max_k']
    proximity_threshold = config['decon_kmeans_proximity_threshold']
    min_cluster_detections = config['decon_kmeans_min_cluster_detections']

    k = 0
    sizes_string = ''
    for i in range(1, max_k + 1):
        #with threadpool_limits(limits=1, user_api="blas"):
        kmeans_version = 'scipy' # 'scipy' or 'sklearn'
        if kmeans_version== 'sklearn':
            kmeans = sklearn.cluster.KMeans(n_clusters=i, random_state=0, n_init=1)
            kmeans.fit(z.reshape(-1, 1))
            cluster_means = kmeans.cluster_centers_.flatten()
            labels = kmeans.labels_
        elif kmeans_version == 'scipy':
            centroids, labels = scipy.cluster.vq.kmeans2(z.reshape(-1, 1), i, minit='++')
            cluster_means = centroids.flatten()
        else:
            raise ValueError(f"Unknown kmeans_version called in deconvolve_kmeans(): {kmeans_version}")
        cluster_sds = np.zeros(i)
        cluster_n = np.zeros(i)
        # compute the size and standard deviation of each cluster
        for j in range(i):
            cluster_sds[j] = np.std(z[labels == j])
            cluster_n[j] = np.sum(labels == j)

        # check if any cluster has fewer than min_cluster_size detections
        if np.any(cluster_n < min_cluster_detections):
            break
        # Order clusters by ascending cluster_means
        sorted_indices = np.argsort(cluster_means)
        cluster_means_sorted = cluster_means[sorted_indices]
        cluster_sds_sorted = cluster_sds[sorted_indices]

        cluster_separations = np.abs(cluster_means_sorted[1:] - cluster_means_sorted[:-1])
        # check if any two clusters are within proximity_threshold * their SDs of each other
        if np.any(cluster_separations < proximity_threshold * (cluster_sds_sorted[1:] + cluster_sds_sorted[:-1])):
            break
        # Save the clustering because this is the one we're using
        k = i
        labels_k = labels.copy()
        cluster_means_k = cluster_means.copy()
        cluster_sds_k = cluster_sds.copy()
        sizes_string_k = '_'.join([str(int(cluster_n[j])) for j in range(k)])

    logging.info(f"Deconvolve - Points: {len(z)} Clusters: {k} Cluster Sizes: {sizes_string_k}")

    if k == 0:
        return z
    # Squeeze each z value to the mean of its cluster
    z_deconv = np.copy(z)
    for i in range(k):
        mask = (labels_k == i)
        # Only shrink the cluster if it is larger than min_cluster_sd
        if cluster_sds_k[i] > min_cluster_sd:
            z_deconv[mask] = cluster_means_k[i] + (z[mask] - cluster_means_k[i]) * sd_shrink_ratio

    if config['debug']:
        decon_summary_stats = os.path.join(config['output_dir'], f'cluster_plots/summary.tsv')
        filename = f"clusters_{k}_points_{len(z)}_{sizes_string_k}"
        plot_deconvolution(z, z_deconv, filename, config)
        # print out the cluster stats for debugging into filename.tsv
        with open(decon_summary_stats, 'a') as f:
            for i in range(k):
                cluster_data = z[labels_k == i]
                f.write(
                    f"{k}\t{np.sum(labels_k == i)}\t{np.mean(cluster_data)}\t{np.std(cluster_data)}\t{np.min(cluster_data)}\t{np.max(cluster_data)}\t{np.median(cluster_data)}\n")
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

def plot_deconvolution(z: np.ndarray, z_adjusted: np.ndarray, filename: str, config: dict):
    logging.info("plot_deconvolution")
    plt.figure()
    plt.hist(z, bins=100, alpha=0.5, label='Before')
    plt.hist(z_adjusted, bins=100, alpha=0.5, label='After')
    plt.legend()
    outdir = os.path.join(config['output_dir'], 'cluster_plots')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f'{filename}.{config['plot_format']}')
    plt.savefig(outfile)
    plt.close()