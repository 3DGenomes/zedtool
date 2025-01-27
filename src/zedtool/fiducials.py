#!/usr/bin/env python3
import numpy as np
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.filters
import skimage.morphology
from scipy.cluster.hierarchy import weighted
from skimage.morphology import binary_dilation, disk
import scipy.ndimage
import scipy.stats
from sklearn.cluster import KMeans
import tifffile
import os
import logging
from zedtool.detections import im_to_detection_entry, median_by_time, fwhm_from_points
from zedtool.plots import plot_histogram, plot_scatter, plotly_scatter
from zedtool.plots import construct_plot_path

def find_fiducials(img: np.ndarray, df: pd.DataFrame, x_idx: np.ndarray, y_idx: np.ndarray, config: dict)  -> Tuple[np.ndarray, np.ndarray]:
    logging.info('find_fiducials')
    # Find fiducials and label them in the detections array

    filling_disc_radius = config['filling_disc_radius']
    median_filter_disc_radius = config['median_filter_disc_radius']
    dilation_disc_radius = config['dilation_disc_radius']
    # gaussian_filter_disc_radius = 1
    # segmentation_classification_plot_file = 'segmentation_classification_plot.png'
    segmentation_mask_file = 'segmentation_mask.tif'
    detections_img_file = 'detections_img.tif'
    fiducial_mask_file = 'fiducial_mask.tif'
    plot_histogram(np.log10(img[img>0]), 'log10(intensity)', 'Fiducials', 'Histogram of log10(intensity) of fiducials', 'histogram_log_intensity', config)
    image_path = os.path.join(config['output_dir'], detections_img_file)
    tifffile.imsave(image_path, img)
    img_filt = skimage.filters.median(img, skimage.morphology.disk(median_filter_disc_radius))
    # img_filt = skimage.filters.gaussian(img, sigma=gaussian_filter_disc_radius)
    if config['only_fiducials']:
        thresh = config['min_fiducial_detections']
    else:
        thresh = skimage.filters.threshold_otsu(img_filt)
    img_mask = img_filt > thresh
    img_mask2 = skimage.morphology.binary_closing(img_mask, footprint=skimage.morphology.disk(filling_disc_radius))
    img_mask3 = scipy.ndimage.binary_fill_holes(img_mask2, skimage.morphology.disk(filling_disc_radius))
    img_mask4 = skimage.morphology.dilation(img_mask3, disk(dilation_disc_radius))
    img_label = skimage.morphology.label(img_mask4)

    image_path = os.path.join(config['output_dir'], segmentation_mask_file)
    tifffile.imsave(image_path, img_label)

    rois = skimage.measure.regionprops_table(img_label, img, properties=('label','bbox', 'centroid', 'area', 'intensity_mean'))
    df_fiducials = pd.DataFrame(rois)
    logging.info(f'Found {len(df_fiducials)} segmented regions before filtering')
    df_fiducials.columns = ['label','min_y', 'min_x', 'max_y', 'max_x', 'centroid_y', 'centroid_x', 'area', 'mean_intensity']
    # Keep brightest regions as decided by the best separation into two groups based on log10(mean_intensity)+1
    df_fiducials['log_intensity'] = np.log10(df_fiducials['mean_intensity']+1)
    kmeans = KMeans(n_clusters=2,  n_init='auto', random_state=0).fit(df_fiducials[['log_intensity']])
    if np.mean(df_fiducials[kmeans.labels_==1]['log_intensity']) > np.mean(df_fiducials[kmeans.labels_==0]['log_intensity']):
        is_high = (kmeans.labels_==1)
    else:
        is_high = (kmeans.labels_==0)
    # If only_fiducials is set to True, keep all fiducials
    if config['only_fiducials']:
        is_high = np.ones(len(df_fiducials), dtype=bool)
    excluded_labels = df_fiducials[is_high==False]['label']
    # scatter plot of log_intensity vs area, with the two clusters colored differently
    plot_scatter(df_fiducials['log_intensity'], df_fiducials['area'], 'log10(mean_intensity+1)', 'area (bins)', 'Segmentation classification', 'segmentation_classification_plot', config)
    df_fiducials = df_fiducials[is_high]
    df_fiducials = df_fiducials.reset_index(drop=True)
    logging.info(f'Found {len(df_fiducials)} segmented regions after filtering')
    # Set pixels in img_label to zero for excluded labels
    labels = img_label.flatten()
    for label in excluded_labels:
        labels[labels==label] = 0
    fiducial_labels = labels.reshape(img_label.shape)
    image_path = os.path.join(config['output_dir'], fiducial_mask_file)
    tifffile.imsave(image_path, fiducial_labels)

    # Use the regions in img_label to label the detections in df
    df['label'] = im_to_detection_entry(fiducial_labels, x_idx, y_idx)
    return df, df_fiducials

def make_fiducial_stats(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> np.ndarray:
    # Make stats for fiducials
    logging.info('make_fiducial_stats')
    # Add columns to df_fiducials for stats
    df_fiducials['n_detections'] = 0
    df_fiducials['n_images'] = 0
    df_fiducials['detections_per_image'] = 0
    df_fiducials['x_mean'] = 0
    df_fiducials['x_sd'] = 0
    df_fiducials['x_fwhm'] = 0
    df_fiducials['y_mean'] = 0
    df_fiducials['y_sd'] = 0
    df_fiducials['y_fwhm'] = 0
    df_fiducials['z_mean'] = 0
    df_fiducials['z_sd'] = 0
    df_fiducials['z_fwhm'] = 0
    df_fiducials['photons_mean'] = 0
    df_fiducials['photons_sd'] = 0
    df_fiducials['name'] = ''
    for j in range(len(df_fiducials)):
        fiducial_label = df_fiducials.at[j,'label']
        logging.info(f'Processing fiducial {fiducial_label}')
        df_sel = df[df['label'] == fiducial_label]
        n_detections = len(df_sel)
        if n_detections == 0:
            logging.error(f'No detections for fiducial {fiducial_label}')
            continue
        x = df_sel[config['x_col']]
        y = df_sel[config['y_col']]
        z = df_sel[config['z_col']]
        x_sd = np.std(x)
        x_mean = np.mean(x)
        x_fwhm = fwhm_from_points(x)
        y_sd = np.std(y)
        y_mean = np.mean(y)
        y_fwhm = fwhm_from_points(y)
        z_sd = np.std(z)
        z_mean = np.mean(z)
        z_fwhm = fwhm_from_points(z)
        # vx, vy, vz are the relative movements per frame
        vx = np.diff(x)
        vy = np.diff(y)
        vz = np.diff(z)
        vx_mad = np.median(np.abs(vx - np.median(vx)))
        vy_mad = np.median(np.abs(vy - np.median(vy)))
        vz_mad = np.median(np.abs(vz - np.median(vz)))
        n_images = len(np.unique(df_sel[config['image_id_col']]))
        if n_images ==0:
            logging.error(f'No images for fiducial {fiducial_label}')
            continue
        detections_per_image = n_detections/n_images
        photons_mean = np.mean(df_sel[config['photons_col']])
        photons_sd = np.std(df_sel[config['photons_col']])
        df_fiducials.at[j, 'n_detections'] = n_detections
        df_fiducials.at[j, 'x_sd'] = x_sd
        df_fiducials.at[j, 'x_mean'] = x_mean
        df_fiducials.at[j, 'y_sd'] = y_sd
        df_fiducials.at[j, 'y_mean'] = y_mean
        df_fiducials.at[j, 'z_sd'] = z_sd
        df_fiducials.at[j, 'z_mean'] = z_mean
        df_fiducials.at[j, 'n_images'] = n_images
        df_fiducials.at[j, 'detections_per_image'] = detections_per_image
        df_fiducials.at[j, 'photons_mean'] = photons_mean
        df_fiducials.at[j, 'photons_sd'] = photons_sd
        df_fiducials.at[j, 'vx_mad'] = vx_mad
        df_fiducials.at[j, 'vy_mad'] = vy_mad
        df_fiducials.at[j, 'vz_mad'] = vz_mad
        df_fiducials.at[j, 'x_fwhm'] = x_fwhm
        df_fiducials.at[j, 'y_fwhm'] = y_fwhm
        df_fiducials.at[j, 'z_fwhm'] = z_fwhm
        df_fiducials.at[j, 'name'] = f'f_{fiducial_label:04d}_z_{int(z_mean):05d}_y_{int(y_mean):05d}_x_{int(x_mean):05d}'.replace('-', 'm')
    return df_fiducials


def filter_fiducials(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Filter fiducials based on stats
    # Plot histograms of stats from n_detections to photons_sd
    logging.info('filter_fiducials')
    # Small and sparse fiducials are likely noise
    detections_cutoff = config['min_fiducial_detections']
    area_cutoff = config['min_fiducial_size']
    doublet_cutoff = config['max_detections_per_image']
    quantile_tail_cutoff = config['quantile_tail_cutoff']
    quantile_max = 1 - quantile_tail_cutoff
    quantile_min = quantile_tail_cutoff

    logging.info(f'Filtering fiducials with fewer than {detections_cutoff} detections or area less than {area_cutoff}')
    logging.info(f'n_fiducials before filtering: {len(df_fiducials)}')
    # Get rid of those that are too small or too sparse
    idx = (
        (df_fiducials['n_detections'] >= detections_cutoff) &
        (df_fiducials['area'] >= area_cutoff)
    )
    if np.sum(idx) == 0:
        logging.error('No fiducials left after filtering for detections and area')
        return None, None
    excluded_labels = df_fiducials[idx==False]['label']
    df_fiducials = df_fiducials[idx]
    df_fiducials = df_fiducials.reset_index(drop=True)
    logging.info(f'n_fiducials after filtering for detections and area: {len(df_fiducials)}')
    # Get rid of those that move or wobble the most
    df_fiducials['vr_mad'] = np.sqrt(df_fiducials['vx_mad']**2 + df_fiducials['vy_mad']**2)
    df_fiducials['r_sd'] = np.sqrt(df_fiducials['x_sd']**2 + df_fiducials['y_sd']**2)
    vr_mad_cutoff = np.quantile(df_fiducials['vr_mad'], quantile_max)
    r_sd_cutoff = np.quantile(df_fiducials['r_sd'], quantile_max)
    photons_sd_cutoff = np.quantile(df_fiducials['photons_sd'], quantile_max)
    photons_mean_cutoff = np.quantile(df_fiducials['photons_mean'], quantile_max)
    n_detections_cutoff = np.quantile(df_fiducials['n_detections'], quantile_min)
    idx = (
        (df_fiducials['vr_mad'] <= vr_mad_cutoff) &
        (df_fiducials['r_sd'] <= r_sd_cutoff) &
        (df_fiducials['photons_sd'] <= photons_sd_cutoff) &
        (df_fiducials['photons_mean'] <= photons_mean_cutoff) &
        (df_fiducials['n_detections'] >= n_detections_cutoff) &
        (df_fiducials['detections_per_image'] <= doublet_cutoff)
    )
    excluded_labels = pd.concat([excluded_labels, df_fiducials[idx == False]['label']])
    # Add to excluded labels the comma separated list in config['exclude_fiducials']
    if 'exclude_fiducials' in config:
        excluded_labels = pd.concat([excluded_labels, pd.Series(config['exclude_fiducials'].split(','))])
        logging.info(f'Excluded labels: {excluded_labels}')

    df_fiducials = df_fiducials[idx]
    df_fiducials = df_fiducials.reset_index(drop=True)
    logging.info(f'n_fiducials after filtering for stability and photons: {len(df_fiducials)}')
    if len(df_fiducials) == 0:
        logging.error('No fiducials left after filtering for stability and photons')
        return None, None

    for col in ['n_detections', 'n_images', 'detections_per_image', 'x_mean', 'x_sd', 'y_mean', 'y_sd', 'z_mean', 'z_sd', 'photons_mean', 'photons_sd', 'area', 'vx_mad', 'vy_mad', 'vz_mad']:
        outpath = os.path.join(config['fiducial_dir'], f"hist_{col}")
        plot_histogram(df_fiducials[col], col, 'Fiducials', '', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "x_sd_vs_y_sd")
    plot_scatter(df_fiducials['x_sd'], df_fiducials['y_sd'], 'x_sd (nm)', 'y_sd (nm)', 'x_sd vs y_sd', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "x_sd_vs_vx_mad")
    plot_scatter(df_fiducials['x_sd'], df_fiducials['vx_mad'], 'x_sd (nm)', 'vx_mad (nm)', 'x_sd vs vx_mad', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "y_sd_vs_vy_mad")
    plot_scatter(df_fiducials['y_sd'], df_fiducials['vy_mad'], 'y_sd (nm)', 'vy_mad (nm)', 'y_sd vs vy_mad', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "vx_mad_vs_vy_mad")
    plot_scatter(df_fiducials['vx_mad'], df_fiducials['vy_mad'], 'vx_mad (nm)', 'vy_mad (nm)', 'vx_mad vs vy_mad', outpath, config)

    # set excluded_labels from df.labels to 0 in df
    df['label'] = df['label'].replace(excluded_labels.tolist(), 0)
    df_filtered = df[df['label'] != 0]
    return  df_filtered, df_fiducials


def plot_fiducial_correlations(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Make correlations between fiducial stats
    logging.info('make_fiducial_correlations')
    n_fiducials = len(df_fiducials)
    n_images = np.max(df[config['image_id_col']]) + 1
    logging.info(f'Making fiducial array: n_fiducials: {n_fiducials}, n_images: {n_images}')
    z =  np.full((n_images, n_fiducials), np.nan)
    for i in range(len(df_fiducials)):
        label = df_fiducials.at[i, 'label']
        idx = df['label'] == label
        frames = df[idx][config['image_id_col']]
        z[frames, i] = df[idx][config['z_col']]
    # make the column names from z_mean column in df_fiducials
    col_z = df_fiducials['z_mean'].astype(int).astype(str).str.zfill(4).str.replace('-', 'm')
    col_y = df_fiducials['y_mean'].astype(int).astype(str).str.zfill(4).str.replace('-', 'm')
    col_x = df_fiducials['x_mean'].astype(int).astype(str).str.zfill(4).str.replace('-', 'm')
    col_label = df_fiducials['label'].astype(int).astype(str).str.zfill(3)
    # colnames = [f"z_{z}_y_{y}_x_{x}" for z, y, x in zip(col_z, col_y, col_x)]
    colnames = [f"z_{z}_f_{fid}" for z, fid in zip(col_z, col_label)]
    logging.info(f'Making array of fiducial distances for {n_fiducials} fiducials')
    dz_mad = np.zeros((n_fiducials, n_fiducials))
    dzdt_mad = np.zeros((n_fiducials, n_fiducials))
    z_cor = np.zeros((n_fiducials, n_fiducials))
    dx = np.zeros((n_fiducials, n_fiducials))
    dy = np.zeros((n_fiducials, n_fiducials))
    dz = np.zeros((n_fiducials, n_fiducials))
    # TODO: Somewhere in here we get this: nanfunctions.py:1217: RuntimeWarning: All-NaN slice encountered
    for i in range(n_fiducials):
        for j in range(i+1, n_fiducials):
            dzij = z[:,i] - z[:,j]
            median_dz = np.nanmedian(dzij)
            dz_mad[i,j] = np.nanmedian(np.abs(dzij - median_dz))
            dzdt = np.diff(dzij)
            median_dzdt = np.nanmedian(dzdt)
            dzdt_mad[i,j] = np.nanmedian(np.abs(dzdt - median_dzdt))
            mask = ~np.isnan(z[:,i]) & ~np.isnan(z[:,j])
            if np.sum(mask) >= 2:
                z_cor[i,j] = scipy.stats.pearsonr(z[mask,i], z[mask,j])[0]
            else:
                z_cor[i,j] = np.nan
            dx[i,j] = np.abs(df_fiducials.at[i, 'x_mean'] - df_fiducials.at[j, 'x_mean'])
            dy[i,j] = np.abs(df_fiducials.at[i, 'y_mean'] - df_fiducials.at[j, 'y_mean'])
            dz[i,j] = np.abs(df_fiducials.at[i, 'z_mean'] - df_fiducials.at[j, 'z_mean'])

    # plot_scatter for all pairs in dx, dy, dz, dz_mad, dzdt_mad
    dimensions = ['x', 'y', 'z']
    distances = [dx, dy, dz]
    plot_quantities = [z_cor, dz_mad, dzdt_mad]
    quantity_names = ['z_cor', 'dz_mad', 'dzdt_mad']
    for quantity, quantity_name in zip(plot_quantities, quantity_names):
        for dim, dist in zip(dimensions, distances):
            xlabel = f'{quantity_name} (nm)'
            ylabel = f'Distance in {dim} (nm)'
            outpath = os.path.join(config['fiducial_dir'], f"fiducial_{quantity_name}_vs_d{dim}")
            idx = (quantity != 0) & (~np.isnan(quantity))
            plot_scatter(quantity[idx], dist[idx], xlabel, ylabel,
                         f'{quantity_name} vs distance', outpath, config)


    # Fill in the distance gaps with the mean for the row and column (or near enough)
    for i in range(n_fiducials):
        for j in range(i+1, n_fiducials):
            if np.isnan(dz_mad[i,j]):
                dz_mad[i,j] = (np.nansum(dz_mad[i,:]) + np.nansum(dz_mad[:,j])) / n_fiducials
            if np.isnan(dzdt_mad[i,j]):
                dzdt_mad[i,j] = (np.nansum(dzdt_mad[i,:]) + np.nansum(dzdt_mad[:,j])) / n_fiducials
            if np.isnan(z_cor[i,j]):
                z_cor[i,j] = (np.nansum(z_cor[i,:]) + np.nansum(z_cor[:,j])) / n_fiducials
    # Make the distance matrix symmetric
    dz_mad = dz_mad + dz_mad.T
    dzdt_mad = dzdt_mad + dzdt_mad.T
    condensed_distance = scipy.spatial.distance.squareform(dz_mad)
    # Perform hierarchical clustering
    z = scipy.cluster.hierarchy.linkage(condensed_distance, method='average')
    plt.figure(figsize=(16, 10))
    scipy.cluster.hierarchy.dendrogram(z, labels=colnames)
    plt.title("Dendrogram (dz_mad)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    outfile = os.path.join(config['fiducial_dir'], "dendrogram_dz_mad")
    figure_path = construct_plot_path(outfile, "png", config)
    plt.savefig(figure_path, dpi=600)
    plt.close()
    # Mask the lower triangle
    mask = np.tri(dz_mad.shape[0], k=-1)
    upper_triangle = np.ma.masked_array(dz_mad, mask=mask)

    # same for dzdt_mad
    condensed_distance = scipy.spatial.distance.squareform(dzdt_mad)
    # Perform hierarchical clustering
    z = scipy.cluster.hierarchy.linkage(condensed_distance, method='average')
    plt.figure(figsize=(16, 10))
    scipy.cluster.hierarchy.dendrogram(z, labels=colnames)
    plt.title("Dendrogram (dzdt_mad)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    outfile = os.path.join(config['fiducial_dir'], "dendrogram_dzdt_mad")
    figure_path = construct_plot_path(outfile, "png", config)
    plt.savefig(figure_path, dpi=600)
    plt.close()
    return dz_mad, dzdt_mad

def make_quality_metrics(df: pd.DataFrame, df_fiducials: pd.DataFrame, config: dict) -> np.ndarray:
    # write csv file with quality metrics for fiducials. One row with columns variance of columns in df_fiducials
    logging.info('make_quality_metrics')
    df_metrics = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0]],
                              columns=['x_sd_mean', 'y_sd_mean', 'z_sd_mean', 'vx_mad_mean', 'vy_mad_mean', 'vz_mad_mean', 'z_fiducial_sd', 'z_non_fiducial_sd'])
    df_metrics['x_sd_mean'] = df_fiducials['x_sd'].mean()
    df_metrics['y_sd_mean'] = df_fiducials['y_sd'].mean()
    df_metrics['z_sd_mean'] = df_fiducials['z_sd'].mean()
    df_metrics['fwhm_x_mean'] = df_fiducials['x_fwhm'].mean()
    df_metrics['fwhm_y_mean'] = df_fiducials['y_fwhm'].mean()
    df_metrics['fwhm_z_mean'] = df_fiducials['z_fwhm'].mean()
    df_metrics['vx_mad_mean'] = df_fiducials['vx_mad'].mean()
    df_metrics['vy_mad_mean'] = df_fiducials['vy_mad'].mean()
    df_metrics['vz_mad_mean'] = df_fiducials['vz_mad'].mean()
    df_metrics['z_fiducial_sd'] = df.loc[df['label']!=0,config['z_col']].std()
    df_metrics['z_non_fiducial_sd'] = df.loc[df['label']==0,config['z_col']].std()
    return df_metrics

def correct_fiducials(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('correct_fiducials')
    # If x1,... are taken then move them to x2,... first.
    # Check if backup column x_0 exists, if not then quit
    if not 'x_0' in df.columns:
        logging.error('No backup columns found in df')
        return df_fiducials, df

    for index, row in df_fiducials.iterrows():
        correct_fiducial(row.to_dict(), df, config)

    return df_fiducials, df

def correct_fiducial(fiducial: dict, df: pd.DataFrame, config: dict) -> int:
    fiducial_label = fiducial['label']
    fiducial_name = fiducial['name']
    logging.info(f'correct_fiducial: {fiducial_name}')
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
    varnames = ['x', 'y', 'z']
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    sd_colnames = [config['x_sd_col'], config['y_sd_col'], config['z_sd_col']]
    # Create array to hold x,y,z for each cycle
    x_ct = np.zeros((total_cycles, frames_per_cycle), dtype=float)
    sd_ct = np.zeros((total_cycles, frames_per_cycle), dtype=float)
    x_ct.fill(np.nan)
    for k, colname in enumerate(xyz_colnames):
        logging.info(f'Correcting {fiducial_label} {fiducial_name}:{colname}')
        # Get x,y,z values for each cycle
        for j in range(num_time_points):
            for i in range(num_cycles):
                idx = (
                        (df['label'] == fiducial_label) &
                    (df[config['cycle_col']] == i + min_cycle) &
                    (df[config['time_point_col']] == j + min_time_point)
                )
                df_sel = df[idx]
                cycle_index = i + j * num_cycles
                frame_index = df_sel[config['frame_col']] - min_frame + (df_sel[config['z_step_col']] - min_z_step) * num_frames
                x_ct[cycle_index, frame_index] = df_sel[colname].values
                sd_ct[cycle_index, frame_index] = df_sel[sd_colnames[k]].values
        dx_c = make_corrections_for_cycles(x_ct, sd_ct, config)
        c_z_step = make_corrections_for_zstep(x_ct, sd_ct, dx_c, config)
        x_ct_cor = apply_corrections_for_zstep(x_ct, c_z_step, config)
        sd_t = estimate_errors_for_zstep(x_ct_cor, config)
        # Plot fitted values on top of original values
        # plot_fiduciual_zstep_fit(fiducial_label,k,x_ct, sd_t, dx_c, c_z_step, config)
        # Transfer corrected values back to df
        for j in range(num_time_points):
            for i in range(num_cycles):
                idx = (
                        (df['label'] == fiducial_label) &
                        (df[config['cycle_col']] == i + min_cycle) &
                    (df[config['time_point_col']] == j + min_time_point)
                )
                cycle_index = i + j * num_cycles
                frame_index = df[idx][config['frame_col']] - min_frame + (df[idx][config['z_step_col']] - min_z_step) * num_frames
                df.loc[idx, colname] = x_ct_cor[cycle_index, frame_index]
                df.loc[idx, sd_colnames[k]] = sd_t[frame_index]

        varname = varnames[k]
        outdir = os.path.join(config['fiducial_dir'], f"{fiducial_name}")
        outpath = os.path.join(outdir, f"{fiducial_name}_cor_{varname}_vs_frame")
        df_sel = df[df['label'] == fiducial_label]
        plot_scatter(df_sel[config['image_id_col']], df_sel[colname], 'image-ID', f'{varname} (nm)', f"{varname} corrected vs frame",
                     outpath, config)
        plotly_scatter(df_sel[config['image_id_col']], df_sel[colname], df_sel[sd_colnames[k]], 'image-ID', f'{varname} (nm)', f"{varname} corrected vs frame",
                       outpath, config)
    return 0

def plot_fiduciual_zstep_fit(fiducial_index: int, dimension_index: int, y: np.ndarray, ysd: np.ndarray, cycle_fit: np.ndarray, z_fit: np.ndarray, config: dict) -> int:
    logging.info('plot_fiduciual_zstep_fit')
    dimensions = ['x', 'y', 'z']
    dim = dimensions[dimension_index]
    outdir = os.path.join(config['output_dir'], "fiducial_zstep_fit")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"f_{fiducial_index}_d_{dim}_fit")
    x = np.arange(len(y))
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(x, y, c = ysd, s = 0.1, label='Original Data')
    plt.colorbar(sc, label='sd')
    # plt.scatter(x, y_fit, s=0.1, label='Fit')
    plt.xlabel('image-ID')
    plt.ylabel(f"{dim} (nm)")
    plt.title(f'Fit for {dim}  fid={fiducial_index} tp={interval_index}')
    plt.legend()
    plt.savefig(outpath)
    plt.close()
    return 0

def make_corrections_for_cycles(x_ct: np.ndarray, sd_ct: np.ndarray, config: dict) -> np.ndarray:
    # Make corrections for cycles - add this number to each cycle of fiducial.
    # Meant to correct for drift during z-step compensation.
    # There may be nan's since not all fiducials have values at all frames
    logging.info('make_corrections_for_cycles')
    x_ct_masked = np.ma.masked_invalid(x_ct)
    sd_ct_masked = np.ma.masked_invalid(sd_ct)
    combined_mask = np.logical_or(x_ct_masked.mask, sd_ct_masked.mask)
    non_zero_mask = sd_ct != 0
    weights = np.full_like(sd_ct, np.nan, dtype=float)
    weights[non_zero_mask] = 1 / sd_ct[non_zero_mask]
    x_ct_masked_combined = np.ma.masked_array(x_ct, mask=combined_mask)
    weights_masked_combined = np.ma.masked_array(weights, mask=combined_mask)
    # Weighted average, ignoring NaNs
    c_cycle = -np.ma.average(x_ct_masked_combined, axis=1, weights=weights_masked_combined).filled(0)
    return c_cycle[:, None]

def make_corrections_for_zstep(x_ct: np.ndarray, sd_ct: np.ndarray, dx_c: np.ndarray, config: dict) -> np.ndarray:
    # Make corrections for zstep
    logging.info('make_corrections_for_zstep')
    min_z_step, max_z_step = map(int, config['z_step_range'].split('-'))
    num_zsteps = max_z_step - min_z_step + 1
    total_frames = x_ct.shape[1]
    frames_per_zstep = int(total_frames / num_zsteps)
    c_z_step = np.zeros(num_zsteps)
    for i in range(num_zsteps):
        x_ct_z = x_ct[:, i * frames_per_zstep:(i + 1) * frames_per_zstep] + dx_c
        sd_ct_z = sd_ct[:, i * frames_per_zstep:(i + 1) * frames_per_zstep]
        x_ct_masked = np.ma.masked_invalid(x_ct_z)
        sd_ct_masked = np.ma.masked_invalid(sd_ct_z)
        combined_mask = np.logical_or(x_ct_masked.mask, sd_ct_masked.mask)
        non_zero_mask = sd_ct_z != 0
        weights = np.full_like(sd_ct_z, np.nan, dtype=float)
        weights[non_zero_mask] = 1 / sd_ct_z[non_zero_mask]
        x_ct_masked_combined = np.ma.masked_array(x_ct_z, mask=combined_mask)
        weights_masked_combined = np.ma.masked_array(weights, mask=combined_mask)
        # Weighted average, ignoring NaNs
        if np.any(~x_ct_masked_combined.mask):
            c_z_step[i] = -np.ma.average(x_ct_masked_combined, weights=weights_masked_combined)
        # TODO: Debugging plots for this z-step correction. Where were the nan's coming from?
    return c_z_step

def apply_corrections_for_zstep(x_ct: np.ndarray, c_z_step: np.ndarray, config: dict) -> np.ndarray:
    num_zsteps = c_z_step.shape[0]
    total_frames = x_ct.shape[1]
    frames_per_zstep = int(total_frames / num_zsteps)

    ct_z_step = np.zeros_like(x_ct)
    for i in range(num_zsteps):
        ct_z_step[:, i * frames_per_zstep:(i + 1) * frames_per_zstep] = c_z_step[i]
    x_ct_cor = x_ct + ct_z_step
    return x_ct_cor

def estimate_errors_for_zstep(x_ct: np.ndarray, config: dict) -> np.ndarray:
    logging.info('estimate_errors_for_zstep')
    valid_counts = np.sum(~np.isnan(x_ct), axis=0)
    idx = valid_counts > 1
    sd_t = np.full(x_ct.shape[1], np.nan)
    sd_t[idx] = np.nanstd(x_ct[:, idx], axis=0)
    return sd_t

def zstep_correction_cost_function(c_z_step: np.ndarray, x_ct: np.ndarray, sd_ct: np.ndarray, dx_c, config: dict) -> float:
    # Calculate the cost function for the corrections
    x_ct_cor = apply_corrections_for_zstep(x_ct, c_z_step, config) + dx_c
    # sd_t = estimate_errors_for_zstep(x_ct_cor, config)
    cost = np.nansum(x_ct_cor**2 / sd_ct**2)
    # print(f'Cost: {cost} sum(c_z_step): {np.sum(c_z_step)}')
    return cost

def correct_detections(df: pd.DataFrame, df_fiducials: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.info('correct_detections')
    noclobber = config['noclobber']
    x_col = ['x', 'y', 'z']
    xsd_col = ['x_sd', 'y_sd', 'z_sd']
    ndims = len(x_col)
    # if fiducials file does not exist, read them in
    outpath_x = os.path.join(config['fiducial_dir'], "fiducials_x.npy")
    outpath_xsd = os.path.join(config['fiducial_dir'], "fiducials_xsd.npy")
    if not os.path.exists(outpath_x) or not noclobber:
        x_ft, xsd_ft = extract_fiducial_detections(df, df_fiducials, config)
        np.save(outpath_x, x_ft)
        np.save(outpath_xsd, xsd_ft)
    else:
        x_ft = np.load(outpath_x)
        xsd_ft = np.load(outpath_xsd)

    # fit fiducials, interpolate across all time points and give uncertainties to interpolated areas
    outpath_x = os.path.join(config['fiducial_dir'], "fiducials_corrected_x.npy")
    outpath_xsd = os.path.join(config['fiducial_dir'], "fiducials_corrected_xsd.npy")
    if not os.path.exists(outpath_x) or not noclobber:
        x_fit_ft, xsd_fit_ft = fit_fiducial_detections(x_ft, xsd_ft, config)
        np.save(outpath_x, x_fit_ft)
        np.save(outpath_xsd, xsd_fit_ft)
    else:
        x_fit_ft = np.load(outpath_x)
        xsd_fit_ft = np.load(outpath_xsd)
    # group fiducials to be zero centred
    x_fit_ft, xsd_fit_ft = group_fiducials(x_fit_ft, xsd_fit_ft, config)
    # Fit drift correction to zero-centred fiducials - including across time-point boundaries
    x_t, x_err = make_corrections(x_fit_ft, xsd_fit_ft, config)
    if config['plot_per_fiducial_fitting']:
        plot_fitted_fiducials(df_fiducials, x_fit_ft, xsd_fit_ft,x_t, config)
    # plot_median_of_detections(df, x_t, config)
    # Save and plot drift correction with error bars
    df_drift = pd.DataFrame({
        'image-ID': np.arange(x_t.shape[1]),
        'x': x_t[0,:], 'x_sd': x_err[0,:], 'y': x_t[1,:], 'y_sd': x_err[1,:], 'z': x_t[2,:], 'z_sd': x_err[2,:]
    })
    output_path = os.path.join(config['output_dir'], "drift_correction.tsv")
    df_drift.to_csv(output_path, sep='\t', index=False)
    for j in range(ndims):
        output_path = os.path.join(config['output_dir'], f"drift_correction_{x_col[j]}")
        plotly_scatter(df_drift['image-ID'], df_drift[x_col[j]], df_drift[xsd_col[j]], 'image-ID', f'{x_col[j]} correction (nm)', 'Drift correction', output_path, config)
        outpath = os.path.join(config['output_dir'], f"cor_{x_col[j]}_vs_time")
        plot_scatter(np.arange(x_t.shape[1]), x_t[j, :], 'image-ID', f'{x_col[j]} correction (nm)',
                     f'Correction {x_col[j]} vs image-ID', outpath, config)
    # correct detections
    df = apply_corrections(df, x_t, config)
    return df

def plot_median_of_detections(df: pd.DataFrame, x_cor: np.ndarray, config: dict) -> np.ndarray:
    logging.info('plot_median_of_detections')
    x_col = ['x', 'y', 'z']
    ndim = len(x_col)
    m = median_by_time(df, config)
    x_med = np.full_like(x_cor, np.nan)
    x_med[:,0:m.shape[1]] = m
    for j in range(ndim):
        outpath = os.path.join(config['output_dir'], f"median_{x_col[j]}_vs_time")
        plot_scatter(np.arange(x_med.shape[1]), x_med[j,:], 'image-ID', f'{x_col[j]} median (nm)',
                     f'Median {x_col[j]} vs image-ID', outpath, config)
    return x_cor

def plot_fitted_fiducials(df_fiducials: pd.DataFrame, x_fit_ft: np.ndarray, xsd_fit_ft: np.ndarray, x_t: np.ndarray, config: dict) -> int:
    logging.info('plot_fitted_fiducials')

    ndims = x_fit_ft.shape[0]
    nfiducials = x_fit_ft.shape[1]
    x_col = ['x', 'y', 'z']
    for j in range(nfiducials):
        fiducial_name = df_fiducials.name[j]
        logging.info(f'Plotting fitted corrections for {fiducial_name}')
        for k in range(ndims):
            outdir = os.path.join(config['fiducial_dir'], f"{fiducial_name}")
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, f"{fiducial_name}_{x_col[k]}_fit_vs_frame")
            plotly_scatter(np.arange(x_fit_ft.shape[2]),
                           x_fit_ft[k,j,:], xsd_fit_ft[k,j,:],
                           'image-ID', f'{x_col[k]} (nm)', f'{x_col[k]} fit vs frame', outpath, config)

    for k in range(ndims):
        logging.info(f'Plotting combined fitted corrections for {x_col[k]}')
        outdir = config['output_dir']
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f"combined_corrections_{x_col[k]}_vs_frame")
        plt.figure(figsize=(10, 6))
        for j in range(nfiducials):
            label = df_fiducials.label[j]
            plt.scatter(np.arange(x_fit_ft.shape[2]), x_fit_ft[k,j,:], s=0.1, label=f'{label}')
        plt.scatter(np.arange(x_fit_ft.shape[2]), x_fit_ft[k,j,:], s=0.2, c='black', label='fit')
        plt.legend()
        plt.xlabel('image-ID')
        plt.ylabel(f"{x_col[k]} (nm)")
        plt.title(f'Fits for {x_col[k]}')
        plt.savefig(outpath)
        plt.close()
    return 0

def make_corrections(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) -> np.ndarray:
    logging.info('make_corrections')
    # Make corrections from fitted fiducials in x_ft
    # weight the fiducials by their uncertainties at each time point
    w = 1 / xsd_ft
    x_t = np.average(x_ft, axis=1, weights=w)
    x_err = np.sqrt(np.average((x_ft - x_t[:, None])**2, axis=1, weights=w))
    # Make the corrections start at zero
    x_t = x_t - x_t[:, [0]]
    return x_t, x_err

def apply_corrections(df: pd.DataFrame, x_t: np.ndarray, config: dict) -> pd.DataFrame:
    logging.info('apply_corrections')
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    ndimensions = len(xyz_colnames)
    for j in range(ndimensions):
        tidx = df[config['image_id_col']]
        df[xyz_colnames[j]] = df[xyz_colnames[j]] - x_t[j, tidx]
    return df

def group_fiducials(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('group_fiducials')
    # Group fiducials to be zero centred
    ndimensions = x_ft.shape[0]
    nfiducials = x_ft.shape[1]
    x_ret = x_ft.copy()
    w = 1 / xsd_ft
    for i in range(nfiducials):
        for j in range(ndimensions):
            # Mask invalid (NaN) values in x_ft and w
            x_ft_masked = np.ma.masked_invalid(x_ft[j, i, :])
            w_masked = np.ma.masked_invalid(w[j, i, :])
            # Combine the masks
            combined_mask = np.ma.mask_or(x_ft_masked.mask, w_masked.mask)
            # Apply the combined mask to both arrays
            x_ft_masked_combined = np.ma.masked_array(x_ft[j, i, :], mask=combined_mask)
            w_masked_combined = np.ma.masked_array(w[j, i, :], mask=combined_mask)

            # Calculate the weighted average, ignoring NaNs
            weighted_average = np.ma.average(x_ft_masked_combined, weights=w_masked_combined)
            weighted_average = np.average(x_ft[j,i,:], weights=w[j,i,:])
            x_ret[j,i,:] = x_ft[j,i,:] - weighted_average
    # Try and group them together closer with a second pass
    # This is a bit of a hack, but it seems to work.
    # TODO: hold x_ret[:,0,:] constant and, for each j, use optimize to find the offsets that minimize the total variance of x[j,:,:]
    for k in range(np.min((nfiducials,10))):
        for i in range(nfiducials):
            for j in range(ndimensions):
                # find the offset that minimises the RMS distance
                not_this_fiducial_index = np.arange(nfiducials) != i
                offset = np.average(x_ret[j, not_this_fiducial_index, :] - x_ret[j, i, :], weights=w[j, not_this_fiducial_index, :])
                x_ret[j, i, :] = x_ret[j, i, :] + offset
                logging.info(f'pass: {k} fiducial: {i} dimension: {j}  offset: {offset}')
    return x_ret, xsd_ft

def fit_fiducial_detections(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('fit_fiducial_detections')
    # Fit fiducials, interpolate across all time points and give uncertainties to interpolated areas
    ndim = x_ft.shape[0]
    nfiducials = x_ft.shape[1]
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
    frames_per_time_point = num_cycles * frames_per_cycle
    fitting_intervals = np.arange(0, total_frames + frames_per_time_point, frames_per_time_point)
    x_fit_ft = np.zeros_like(x_ft)
    x_fit_ft.fill(np.nan)
    xsd_fit_ft = np.zeros_like(xsd_ft)
    xsd_fit_ft.fill(np.nan)
    for i in range(nfiducials):
        for k in range(ndim):
            # loop over fitting intervals, fitting each interval separately
            for j in range(len(fitting_intervals)-1):
                idx = np.arange(fitting_intervals[j], fitting_intervals[j+1])
                x_fit_ft[k,i,idx], xsd_fit_ft[k,i,idx] = fit_fiducial_step(x_ft[k,i,idx], xsd_ft[k,i,idx], config)
                y = x_ft[k,i,idx]
                ysd = xsd_ft[k,i,idx]
                y_fit = x_fit_ft[k,i,idx]
                # If there are no non-nan values in the interval, skip plotting
                if np.sum(~np.isnan(y)) == 0 or np.sum(~np.isnan(ysd)) == 0 or np.sum(~np.isnan(y_fit)) == 0:
                    logging.warning(f'No valid data for fitting in fit_fiducial_detections() for fiducial {i} dimension {k} interval {j}')
                    # if no fitting possible then assign the last value of the previous fitting interval to this one
                    if j > 0:
                        x_fit_ft[k,i,idx] = x_fit_ft[k,i,fitting_intervals[j]-1]
                        xsd_fit_ft[k,i,idx] = xsd_fit_ft[k,i,fitting_intervals[j]-1]
                    else:
                        # zeros
                        x_fit_ft[k,i,idx] = 0
                        xsd_fit_ft[k,i,idx] = 0
                        continue
                elif config['plot_per_fiducial_fitting']:
                    plot_fiduciual_step_fit(i, j, k, x_ft[k,i,idx], xsd_ft[k,i,idx], x_fit_ft[k,i,idx], xsd_fit_ft[k,i,idx], config)

    # TODO: Leave the zeros in the data and then revisit here and interpolate between adjacent non-zero values from adjacent fitting intervals
    # Prevent possibly divide by zero errors later by replacing zeros with the next non-zero value
    # This will fix runs of zeros in the data that start from the beginning
    for i in range(nfiducials):
        for k in range(ndim):
            # Count down with j to avoid overwriting the dummy values
            for j in range(len(fitting_intervals)-1, 0, -1):
                idx = np.arange(fitting_intervals[j-1], fitting_intervals[j])
                if np.sum(xsd_fit_ft[k,i,idx] == 0) > 0:
                    xsd_fit_ft[k, i, idx] = xsd_fit_ft[k,i,fitting_intervals[j]]
                    x_fit_ft[k, i, idx] = x_fit_ft[k,i,fitting_intervals[j]]
    return x_fit_ft, xsd_fit_ft

def plot_fiduciual_step_fit(fiducial_index: int, interval_index: int, dimension_index: int, y: np.ndarray, ysd: np.ndarray, y_fit: np.ndarray, ysd_fit: np.ndarray, config: dict) -> int:
    logging.info('plot_fiduciual_step_fit')
    dimensions = ['x', 'y', 'z']
    dim = dimensions[dimension_index]
    outdir = os.path.join(config['output_dir'], "fiducial_step_fit")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"fidx_{fiducial_index}_i_{interval_index}_d_{dim}_fit")
    x = np.arange(len(y))
    plt.figure(figsize=(10, 6))
    if np.sum(~np.isnan(y)) == 0 or np.sum(~np.isnan(ysd)) == 0 or np.sum(~np.isnan(y_fit)) == 0:
        logging.warning(f'No valid data for fitting in plot_fiduciual_step_fit() for fiducial {fiducial_index} dimension {dim} interval {interval_index}')
        return
    sc = plt.scatter(x, y, c = ysd, s = 0.1, label='Original Data')
    plt.colorbar(sc, label='sd')
    plt.scatter(x, y_fit+ysd_fit, s=0.1, label='fit+sd')
    plt.scatter(x, y_fit-ysd_fit, s=0.1, label='fit-sd')
    plt.scatter(x, y_fit, s=0.1, label='fit')
    plt.xlabel('image-ID')
    plt.ylabel(f"{dim} (nm)")
    plt.title(f'Fit for {dim}  fid={fiducial_index} tp={interval_index}')
    plt.legend()
    plt.savefig(outpath)
    plt.close()
    return 0

def fit_fiducial_step(xt: np.ndarray, xt_sd: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('fit_fiducial_step')
    polynomial_degree = config['polynomial_degree']
    use_weights_in_fit = (config['use_weights_in_fit']!=0)
    extrapolate_to_end = True
    median_filter_size = 100
    outlier_threshold = 3

    w = 1 / xt_sd
    non_nan_indices = ~np.isnan(xt) & ~np.isnan(w) & ~np.isinf(w)
    nan_indices = ~non_nan_indices
    if np.sum(non_nan_indices) == 0:
        logging.warning('No valid data for fitting in fit_fiducial_step()')
        return np.full_like(xt, np.nan), np.full_like(xt, np.nan)

    first_non_nan = np.min(np.where(non_nan_indices))
    last_non_nan = np.max(np.where(non_nan_indices))
    x = np.arange(len(xt))
    y = xt

    # Fit a polynomial using weights
    if use_weights_in_fit:
        coefficients = np.polyfit(x[non_nan_indices], y[non_nan_indices], polynomial_degree, w=w[non_nan_indices])
    else:
        coefficients = np.polyfit(x[non_nan_indices], y[non_nan_indices], polynomial_degree)
    x_fit = np.polyval(coefficients, x)

    # Calculate error bars of the fit for later use when weighting fits
    residuals = np.abs(y - x_fit)
    outlier_sd = np.nanmean(residuals[non_nan_indices]) * outlier_threshold
    residuals_filled = np.where(nan_indices, outlier_sd, residuals)
    smoothed_residuals = scipy.ndimage.median_filter(residuals_filled, size=median_filter_size)
    xsd_fit = np.copy(smoothed_residuals)
    xsd_fit[xsd_fit>outlier_threshold | np.isnan(residuals)] = outlier_sd

    if not extrapolate_to_end:
        if first_non_nan > 0:
            x_fit[:first_non_nan] = x_fit[first_non_nan]
            xsd_fit[:first_non_nan] = np.nanmean(xsd_fit[non_nan_indices]) * 3
        if last_non_nan < len(x_fit):
            x_fit[last_non_nan:] = x_fit[last_non_nan]
            xsd_fit[last_non_nan:] = np.nanmean(xsd_fit[non_nan_indices]) * 3

    return x_fit, xsd_fit


def extract_fiducial_detections(df: pd.DataFrame, df_fiducials: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Extract fiducials from the DataFrame and put the x,y,z detections into rows of a numpy array
    # array rows are fiducials, array columns are image-ID, encompassing the whole range of image-IDs
    logging.info('extract_fiducial_detections')
    nfiducials = len(df_fiducials)
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
    if np.max(df[config['image_id_col']]) > total_frames:
        logging.error('Image-ID exceeds total number of frames')
        return None, None

    x_ft = np.zeros((3,nfiducials, total_frames), dtype=float)
    x_ft.fill(np.nan)
    xsd_ft = np.zeros((3,nfiducials, total_frames), dtype=float)
    xsd_ft.fill(np.nan)
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    sd_colnames = [config['x_sd_col'], config['y_sd_col'], config['z_sd_col']]
    for i in range(nfiducials):
        label = df_fiducials.at[i, 'label']
        logging.info(f'Extracting detections for fiducial {label}')
        for k, colname in enumerate(xyz_colnames):
            df_sel = df[df['label'] == label]
            image_id = df_sel[config['image_id_col']]
            x_ft[k,i, image_id] = df_sel[colname].values
            xsd_ft[k,i, image_id] = df_sel[sd_colnames[k]].values
    return x_ft, xsd_ft

def find_boundaries(df: pd.DataFrame, colname: str) -> np.ndarray:
    """
    Finds the boundaries in a DataFrame column where the value in colname changes.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to analyze.
    colname (str): The name of the column to find boundaries in.

    Returns:
    np.ndarray: An array of indices representing the boundaries where the column value changes.
    """
    boundaries = np.where(np.diff(df[colname]))[0] + 1
    boundaries = np.insert(boundaries, 0, 0)
    boundaries = np.append(boundaries, df.shape[0]-1)
    return boundaries

