#!/usr/bin/env python3
import numpy as np
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.filters
import skimage.morphology
from skimage.morphology import binary_dilation, disk
import scipy.ndimage
import scipy.stats
from sklearn.cluster import KMeans
import tifffile
import os
import logging
from zedtool.detections import im_to_detection_entry
from zedtool.plots import plot_histogram, plot_scatter

def find_fiducials(img: np.ndarray, df: pd.DataFrame, x_idx: np.ndarray, y_idx: np.ndarray, config: dict)  -> Tuple[np.ndarray, np.ndarray]:
    logging.info('find_fiducials')
    # Find fiducials and label them in the detections array

    median_disc_radius = 1
    dilation_disc_radius = 5
    filling_disc_radius = 5
    gaussian_disc_radius = 1
    segmentation_classification_plot_file = 'segmentation_classification_plot.png'
    segmentation_mask_file = 'segmentation_mask.tif'
    detections_img_file = 'detections_img.tif'
    fiducial_mask_file = 'fiducial_mask.tif'

    image_path = os.path.join(config['output_dir'], detections_img_file)
    tifffile.imsave(image_path, img)
    img_filt = skimage.filters.median(img, skimage.morphology.disk(median_disc_radius))
    # img_filt = skimage.filters.gaussian(img, sigma=gaussian_disc_radius)
    img_mask = img_filt > skimage.filters.threshold_otsu(img_filt)
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
    plot_scatter(df_fiducials['log_intensity'], df_fiducials['area'], 'log10(mean_intensity+1)', 'area', 'Segmentation classification', 'segmentation_classification_plot', config)
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
    df_fiducials['y_mean'] = 0
    df_fiducials['y_sd'] = 0
    df_fiducials['z_mean'] = 0
    df_fiducials['z_sd'] = 0
    df_fiducials['photons_mean'] = 0
    df_fiducials['photons_sd'] = 0

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
        y_sd = np.std(y)
        y_mean = np.mean(y)
        z_sd = np.std(z)
        z_mean = np.mean(z)
        # vx, vy, vz are the relative movements per frame
        vx = np.diff(x)
        vy = np.diff(y)
        vz = np.diff(z)
        vx_mad = np.median(np.abs(vx - np.median(vx)))
        vy_mad = np.median(np.abs(vy - np.median(vy)))
        vz_mad = np.median(np.abs(vz - np.median(vz)))
        n_images = len(np.unique(df_sel[config['frame_col']]))
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
    outpath = os.path.join(config['fiducial_dir'], "fiducials_unfiltered.tsv")
    df_fiducials.to_csv(outpath, sep='\t', index=False)
    return df_fiducials


def filter_fiducials(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Filter fiducials based on stats
    # Plot histograms of stats from n_detections to photons_sd
    logging.info('filter_fiducials')
    # Small and sparse fiducials are likely noise
    detections_cutoff = 100 # TODO: put this in config
    area_cutoff = 10 # TODO: put this in config
    doublet_cutoff = 1.5 # TODO: put this in config
    logging.info(f'Filtering fiducials with fewer than {detections_cutoff} detections or area less than {area_cutoff}')
    logging.info(f'n_fiducials before filtering: {len(df_fiducials)}')
    # Get rid of those that are too small or too sparse
    idx = (
        (df_fiducials['n_detections'] > detections_cutoff) &
        (df_fiducials['area'] > area_cutoff)
    )
    excluded_labels = df_fiducials[idx==False]['label']
    df_fiducials = df_fiducials[idx]
    df_fiducials = df_fiducials.reset_index(drop=True)
    logging.info(f'n_fiducials after filtering for detectinos and area: {len(df_fiducials)}')
    # Get rid of those that move or wobble the most
    df_fiducials['vr_mad'] = np.sqrt(df_fiducials['vx_mad']**2 + df_fiducials['vy_mad']**2)
    df_fiducials['r_sd'] = np.sqrt(df_fiducials['x_sd']**2 + df_fiducials['y_sd']**2)
    vr_mad_cutoff = np.quantile(df_fiducials['vr_mad'], 0.95)
    r_sd_cutoff = np.quantile(df_fiducials['r_sd'], 0.95)
    photons_sd_cutoff = np.quantile(df_fiducials['photons_sd'], 0.95)
    photons_mean_cutoff = np.quantile(df_fiducials['photons_mean'], 0.95)
    n_detections_cutoff = np.quantile(df_fiducials['n_detections'], 0.05)
    idx = (
        (df_fiducials['vr_mad'] < vr_mad_cutoff) &
        (df_fiducials['r_sd'] < r_sd_cutoff) &
        (df_fiducials['photons_sd'] < photons_sd_cutoff) &
        (df_fiducials['photons_mean'] < photons_mean_cutoff) &
        (df_fiducials['n_detections'] > n_detections_cutoff) &
        (df_fiducials['detections_per_image'] < doublet_cutoff)
    )
    excluded_labels = excluded_labels.append(df_fiducials[idx==False]['label'])
    df_fiducials = df_fiducials[idx]
    df_fiducials = df_fiducials.reset_index(drop=True)
    logging.info(f'n_fiducials after filtering for stability and photons: {len(df_fiducials)}')

    for col in ['n_detections', 'n_images', 'detections_per_image', 'x_mean', 'x_sd', 'y_mean', 'y_sd', 'z_mean', 'z_sd', 'photons_mean', 'photons_sd', 'area', 'vx_mad', 'vy_mad', 'vz_mad']:
        outpath = os.path.join(config['fiducial_dir'], f"hist_{col}")
        plot_histogram(df_fiducials[col], col, 'Detections', '', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "x_sd_vs_y_sd")
    plot_scatter(df_fiducials['x_sd'], df_fiducials['y_sd'], 'x_sd', 'y_sd', 'x_sd vs y_sd', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "x_sd_vs_vx_mad")
    plot_scatter(df_fiducials['x_sd'], df_fiducials['vx_mad'], 'x_sd', 'vx_mad', 'x_sd vs vx_mad', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "y_sd_vs_vy_mad")
    plot_scatter(df_fiducials['y_sd'], df_fiducials['vy_mad'], 'y_sd', 'vy_mad', 'y_sd vs vy_mad', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "vx_mad_vs_vy_mad")
    plot_scatter(df_fiducials['vx_mad'], df_fiducials['vy_mad'], 'vx_mad', 'vy_mad', 'vx_mad vs vy_mad', outpath, config)

    # set excluded_labels from df.labels to 0 in df
    df['label'] = df['label'].replace(excluded_labels.tolist(), 0)
    df_filtered = df[df['label'] != 0]
    outpath = os.path.join(config['fiducial_dir'], "fiducials_filtered.tsv")
    df_fiducials.to_csv(outpath, sep='\t', index=False)
    outpath = os.path.join(config['output_dir'], "detections_filtered_fiducuals.csv")
    df_filtered.to_csv(outpath, index=False)
    return  df_filtered, df_fiducials


def make_fiducial_correlations(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Make correlations between fiducial stats
    logging.info('make_fiducial_correlations')
    n_fiducials = len(df_fiducials)
    n_images = np.max(df[config['frame_col']]) + 1
    logging.info(f'Making fiducial array: n_fiducials: {n_fiducials}, n_images: {n_images}')
    z =  np.full((n_images, n_fiducials), np.nan)
    for i in range(len(df_fiducials)):
        label = df_fiducials.at[i, 'label']
        idx = df['label'] == label
        frames = df[idx][config['frame_col']]
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
    outpath = os.path.join(config['fiducial_dir'], "fiducial_z_cor_vs_dz")
    plot_scatter(z_cor[(z_cor!=0) & (~np.isnan(z_cor))], dz[(z_cor!=0) & (~np.isnan(z_cor))], 'z_cor', 'dz', 'z_cor vs dz', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "fiducial_dz_mad_vs_dz")
    plot_scatter(dz_mad[(dz_mad!=0) & (~np.isnan(dz_mad))], dz[(dz_mad!=0) & (~np.isnan(dz_mad))], 'dz_mad', 'dz', 'dz_mad vs dz', outpath, config)
    outpath = os.path.join(config['fiducial_dir'], "fiducial_dzdt_mad_vs_dz")
    plot_scatter(dzdt_mad[(dzdt_mad!=0) & (~np.isnan(dzdt_mad))], dz[(dzdt_mad!=0) & (~np.isnan(dzdt_mad))], 'dzdt_mad', 'dz', 'dzdt_mad vs dz', outpath, config)

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
    outfile = os.path.join(config['fiducial_dir'], "dendrogram_dz_mad.png")
    plt.savefig(outfile, dpi=600)
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
    outfile = os.path.join(config['fiducial_dir'], "dendrogram_dzdt_mad.png")
    plt.savefig(outfile, dpi=600)
    plt.close()


    return dz_mad, dzdt_mad


def correct_fiducials(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    return df_fiducials, df

#def find_cycle_boundaries(df: pd.DataFrame) -> np.ndarray:
#    cycle_boundaries = find_cycle_boundaries(df)
#    cycle_intervals = np.column_stack((cycle_boundaries[:-1], cycle_boundaries[1:]))

