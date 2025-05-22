import numpy as np
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage
import scipy.stats
import os
import logging

def make_time_point_metrics(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Make time point metrics for fiducials
    logging.info('make_time_point_metrics')
    nfiducials = len(df_fiducials)
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    ndims = len(xyz_colnames)
    time_point_col = config['time_point_col']
    min_time_point, max_time_point = map(int, config['time_point_range'].split('-'))
    num_time_points = max_time_point - min_time_point + 1
    # Create a 4D array for the time point metrics
    metrics_ijfd = np.zeros((num_time_points, num_time_points, nfiducials, ndims), dtype=np.float32)
    # Iterate over fiducials, dimensions, and time points
    for fiducial_idx in range(nfiducials):
        fiducial_label = df_fiducials.at[fiducial_idx,'label']
        logging.info(f'Making time point metrics for fiducial {fiducial_label}')
        df_sel = df[df['label'] == fiducial_label]
        n_detections = len(df_sel)
        if n_detections == 0:
            logging.warning(f'No detections for fiducial {fiducial_label}')
            continue
        for dim in range(ndims):
            dimcol = xyz_colnames[dim]
            # i loops from min_time_point to max_time_point
            for i in range(min_time_point, max_time_point + 1):
                idx = df_sel[time_point_col] == i
                if np.sum(idx) == 0:
                    logging.warning(f'No detections for fiducial {fiducial_label} at time point {i}')
                    continue
                x_i = np.nanmean(df_sel[idx][dimcol].values)
                # j loops from min_time_point to max_time_point
                for j in range(i+1, max_time_point + 1):
                    idx = df_sel[time_point_col] == j
                    if np.sum(idx) == 0:
                        logging.warning(f'No detections for fiducial {fiducial_label} at time point {j}')
                        continue
                    x_j = np.nanmean(df_sel[idx][dimcol].values)
                    metrics_ijfd[i,j,fiducial_idx,dim] = np.abs(x_i - x_j)
                    metrics_ijfd[j,i,fiducial_idx,dim] = metrics_ijfd[i,j,fiducial_idx,dim]

    metrics_ifd = np.zeros((num_time_points-1, nfiducials, ndims), dtype=np.float32)
    # metrics_ifd gets the mean of the off-diagonal elements distance i from the diagonal
    for fiducial_idx in range(nfiducials):
        for dim in range(ndims):
            metrics_ij = metrics_ijfd[:, :, fiducial_idx, dim]
            for j in range(0, num_time_points - 1):
                vj = np.diag(metrics_ij, k=j+1)
                metrics_ifd[j, fiducial_idx, dim] = np.nanmean(vj)

    # Make Euclidean distance out of last index
    metrics_ijf = np.sqrt(np.sum(metrics_ijfd**2, axis=3))
    metrics_if = np.sqrt(np.sum(metrics_ifd**2, axis=2))
    # i,j label time points, f label fiducials and d labels dimensions
    return metrics_ijf, metrics_ifd, metrics_if


def plot_time_point_metrics(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> None:
    # Plot fiducials positions and stats relating to their stability over time
    logging.info('plot_time_point_metrics')
    nfiducials = len(df_fiducials)
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col'], 'r']
    ndims_ex = len(xyz_colnames)
    min_time_point, max_time_point = map(int, config['time_point_range'].split('-'))
    num_time_points = max_time_point - min_time_point + 1
    if num_time_points<=1:
        logging.info('Not enough time points to plot')
        return
    # colours for fiducial plots
    colors = cm.get_cmap('tab20', nfiducials)
    # i,j label time points, f label fiducials and d labels dimensions
    metrics_ijf, metrics_ifd, metrics_if = make_time_point_metrics(df_fiducials, df, config)
    metrics_id_median = np.median(metrics_ifd, axis=1)
    metrics_id_mad = scipy.stats.median_abs_deviation(metrics_ifd, axis=1)
    metrics_i_median = np.median(metrics_if, axis=1)
    metrics_i_mad = scipy.stats.median_abs_deviation(metrics_if, axis=1)

    metrics_idex_median = np.concatenate((metrics_id_median, metrics_i_median[:, np.newaxis]), axis=1)
    metrics_idex_mad = np.concatenate((metrics_id_mad, metrics_i_mad[:, np.newaxis]), axis=1)
    metrics_ifdex = np.concatenate((metrics_ifd, metrics_if[:, :, np.newaxis]), axis=2)

    # Plot the time point metrics. metrics_if and metrics_ifd - per fiducial and summary
    # output_file = os.path.join(config['output_dir'], 'time_point_metrics.png')
    for k in range(ndims_ex):
        outdir = config['output_dir']
        os.makedirs(outdir, exist_ok=True)
        y_col = xyz_colnames[k]
        outpath = os.path.join(outdir, f"fiducial_dist_{y_col}_vs_timepoint_dist.png")
        plt.figure(figsize=(10, 6))
        for j in range(nfiducials):
            label = df_fiducials.label[j]
            plt.scatter(np.arange(metrics_ifdex.shape[0])+1, metrics_ifdex[:,j,k], label=f'{label}', color=colors(j))
        plt.scatter(np.arange(metrics_idex_median.shape[0])+1, metrics_idex_median[:,k], c='black', label='median')
        plt.legend(markerscale=0.5, handletextpad=0.1, loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True,
                   framealpha=1, fontsize='small')
        plt.xlabel('time point difference')
        plt.ylabel(f"{y_col} (nm)")
        plt.title(f'Drift as a function of time-point separation')
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()

    for k in range(ndims_ex):
        outdir = config['output_dir']
        os.makedirs(outdir, exist_ok=True)
        y_col = xyz_colnames[k]
        outpath = os.path.join(outdir, f"summary_fiducial_dist_{y_col}_vs_timepoint_dist.png")
        plt.figure(figsize=(10, 6))
        # plt.errorbar(np.arange(metrics_idex_median.shape[0])+1, metrics_idex_median[:,k],  metrics_idex_mad[:,k], linestyle='none', fmt='o', capsize=5)

        # Do a box and whisker plot using metrics_ifdex
        plt.boxplot(np.transpose(metrics_ifdex[:, :, k]), positions=np.arange(metrics_ifdex.shape[0])+1, widths=0.5)
        plt.xlabel('time point difference')
        plt.ylabel(f"{y_col} (nm)")
        plt.title(f'Drift as a function of time-point separation')
        plt.savefig(outpath)
        plt.close()
    # Save metrics_ifdex() to tsv files - one for each entry in xyz_colnames
    for k in range(ndims_ex):
        outdir = config['output_dir']
        os.makedirs(outdir, exist_ok=True)
        y_col = xyz_colnames[k]
        outpath = os.path.join(outdir, f"fiducial_dist_{y_col}_vs_timepoint_dist.tsv")
        row_indices = np.arange(1, metrics_ifdex.shape[0] + 1).reshape(-1, 1)
        data_to_save = np.hstack((row_indices, metrics_ifdex[:, :, k]))
        np.savetxt(outpath, data_to_save, delimiter='\t',
                   header='time_point_difference\t' + '\t'.join(df_fiducials.name), comments='')





