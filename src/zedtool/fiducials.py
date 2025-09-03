import numpy as np
from typing import Tuple
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.filters
import skimage.morphology
from scipy.cluster.hierarchy import weighted
from skimage.morphology import disk
import scipy.ndimage
import scipy.stats
import scipy.ndimage
from sklearn.cluster import KMeans
import os
import platform
import logging
import multiprocessing
from zedtool.detections import im_to_detection_entry, fwhm_from_points, apply_corrections
from zedtool.plots import plot_histogram, plot_scatter, plotly_scatter
from zedtool.plots import construct_plot_path, plot_drift_correction
from zedtool.parallel import minimize_fiducial_fit_variance_parallel
from zedtool.timepoints import make_time_point_metrics
from zedtool.image import add_axes_and_scale_bar

def find_fiducials(img: np.ndarray, df: pd.DataFrame, x_idx: np.ndarray, y_idx: np.ndarray, config: dict)  -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info('find_fiducials')
    # Find fiducials and label them in the detections array
    filling_disc_radius = config['filling_disc_radius']
    median_filter_disc_radius = config['median_filter_disc_radius']
    dilation_disc_radius = config['dilation_disc_radius']
    # gaussian_filter_disc_radius = 1
    fiducial_label_file = 'fiducials_labels.png'
    plot_histogram(np.log10(img[img>0]), 'log10(bin)', 'Number of bins', 'Histogram of binned detections image', 'histogram_binned_detections', config)
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
    if np.sum(img_label) == 0:
        logging.error('No regions found in segmentation.')
        if config['only_fiducials']:
            logging.error(f'Try lowering min_fiducial_detections to less than {np.max(img_filt)}.')
        raise RuntimeError('No fiducials found.')

    rois = skimage.measure.regionprops_table(img_label, img, properties=('label','bbox', 'centroid', 'area', 'intensity_mean'))
    df_fiducials = pd.DataFrame(rois)
    logging.info(f'Found {len(df_fiducials)} segmented regions before filtering')
    df_fiducials.columns = ['label','min_y', 'min_x', 'max_y', 'max_x', 'centroid_y', 'centroid_x', 'area', 'mean_intensity']
    # Keep brightest regions as decided by the best separation into two groups based on log10(mean_intensity)+1
    df_fiducials['log_intensity'] = np.log10(df_fiducials['mean_intensity']+1)
    # Filter fiducials based on clustering on log_intensity - keep the brighter ones
    # Possibly this is best done in filter_fiducials()
    filter_by_clustering = config['filter_fiducials_with_clustering']
    if filter_by_clustering:
        kmeans = KMeans(n_clusters=2,  n_init='auto', random_state=0).fit(df_fiducials[['log_intensity']])
        if np.mean(df_fiducials[kmeans.labels_==1]['log_intensity']) > np.mean(df_fiducials[kmeans.labels_==0]['log_intensity']):
            is_high = (kmeans.labels_==1)
        else:
            is_high = (kmeans.labels_==0)
    # If only_fiducials is set to True, keep all fiducials
    if config['only_fiducials'] or filter_by_clustering == False:
        is_high = np.ones(len(df_fiducials), dtype=bool)
    else:
        logging.info(f'Keeping {np.sum(is_high)} after clustering on log_intensity')

    # Exclude fiducials in config['excluded_fiducials'] by setting is_high to False
    excluded_fiducials =  config['excluded_fiducials']
    included_fiducials =  config['included_fiducials']

    # Reject any fiducials that are in excluded_fiducials and include any that are in included_fiducials
    if len(excluded_fiducials) > 0:
        is_high = is_high & ~df_fiducials['label'].isin(excluded_fiducials)
    if len(included_fiducials) > 0:
        is_high = is_high & df_fiducials['label'].isin(included_fiducials)

    # Set excluded_labels from df_fiducials.labels to 0 in df
    excluded_labels = df_fiducials[is_high==False]['label']
    # scatter plot of log_intensity vs area
    # plot_scatter(df_fiducials['log_intensity'], df_fiducials['area'], 'log10(mean_intensity+1)', 'area (bins)', 'Segmentation classification', 'segmentation_classification_plot', config)
    df_fiducials = df_fiducials[is_high]
    df_fiducials = df_fiducials.reset_index(drop=True)
    logging.info(f'Found {len(df_fiducials)} segmented regions after filtering on clustering and excluded_fiducials')
    # Set pixels in img_label to zero for excluded labels
    labels = img_label.flatten()
    img_is_fiducial = (img_label != 0)
    for label in excluded_labels:
        labels[labels==label] = 0
    fiducial_labels = labels.reshape(img_label.shape)
    # Make png version of filtered fiducial labels
    image_path = os.path.join(config['output_dir'], fiducial_label_file)
    img_label_filtered = 255 * (np.log10(fiducial_labels + 1) / np.log10(np.max(fiducial_labels) + 1))
    img_label_filtered = img_label_filtered.astype(np.uint8)
    imp = Image.fromarray(img_label_filtered)
    imp = add_axes_and_scale_bar(imp, scale_bar_length=50, bin_resolution=config['bin_resolution'])
    imp.save(image_path, quality=95)

    # Use the regions in img_label to label the detections in df
    df['label'] = im_to_detection_entry(fiducial_labels, x_idx, y_idx)
    df['is_fiducial'] = im_to_detection_entry(img_is_fiducial, x_idx, y_idx)
    return df, df_fiducials

def make_fiducial_stats(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Make stats for fiducials
    logging.info('make_fiducial_stats')
    n_fiducials = len(df_fiducials)
    min_time_point, max_time_point = map(int, config['time_point_range'].split('-'))
    num_time_points = max_time_point - min_time_point + 1
    if num_time_points>1:
        metrics_ijfd, metrics_ijf, metrics_ifd, metrics_if = make_time_point_metrics(df_fiducials, df, config)

    for j in range(n_fiducials):
        fiducial_label = df_fiducials.at[j,'label']
        logging.info(f'Making stats for fiducial {fiducial_label}')
        df_sel = df[df['label'] == fiducial_label]
        n_detections = len(df_sel)
        if n_detections == 0:
            logging.error(f'No detections for fiducial {fiducial_label}')
            continue
        x = df_sel[config['x_col']]
        y = df_sel[config['y_col']]
        z = df_sel[config['z_col']]
        deltaz = df_sel[config['deltaz_col']]
        z_step = df_sel[config['z_step_col']]
        # Check if there are enough detections to calculate correlations and if x,y,z,deltaz contain NaNs or inf
        if np.sum(np.isnan(x)) > 0 or np.sum(np.isnan(y)) > 0 or np.sum(np.isnan(z)) > 0:
            logging.error(f'Nans in x,y,z for fiducial {fiducial_label}')
            continue
        if np.sum(np.sum(np.isnan(deltaz))) > 0:
            logging.warning(f'All Nans in deltaz for fiducial {fiducial_label} in make_fiducial_stats()')
            continue
        x_z_step_cor = scipy.stats.pearsonr(x, z_step)[0]
        y_z_step_cor = scipy.stats.pearsonr(y, z_step)[0]
        z_z_step_cor = scipy.stats.pearsonr(z, z_step)[0]
        # fit a linear model of z versus deltaz
        x_deltaz_slope, intercept, x_deltaz_cor, p_value, std_err = scipy.stats.linregress(deltaz, x)
        y_deltaz_slope, intercept, y_deltaz_cor, p_value, std_err = scipy.stats.linregress(deltaz, y)
        z_deltaz_slope, intercept, z_deltaz_cor, p_value, std_err = scipy.stats.linregress(deltaz, z)
        x_sd = np.std(x)
        x_mean = np.mean(x)
        x_fwhm = fwhm_from_points(x)
        y_sd = np.std(y)
        y_mean = np.mean(y)
        y_fwhm = fwhm_from_points(y)
        z_sd = np.std(z)
        z_mean = np.mean(z)
        z_fwhm = fwhm_from_points(z)
        # vx, vy, vz are the 1st derivative wrt time of x, y, z
        vx = np.diff(x)
        vy = np.diff(y)
        vz = np.diff(z)
        # Median Absolute Displacement Rate - median "speed" of fiducial in x, y, z
        x_madr = np.median(np.abs(vx))
        y_madr = np.median(np.abs(vy))
        z_madr = np.median(np.abs(vz))
        n_images = len(np.unique(df_sel[config['image_id_col']]))
        if n_images ==0:
            logging.error(f'No images for fiducial {fiducial_label}')
            continue
        detections_per_image = n_detections/n_images
        photons_mean = np.mean(df_sel[config['photons_col']])
        photons_sd = np.std(df_sel[config['photons_col']])
        df_fiducials.at[j, 'name'] = f'f_{fiducial_label:04d}_z_{int(z_mean):05d}_y_{int(y_mean):05d}_x_{int(x_mean):05d}'.replace('-', 'm')
        df_fiducials.at[j, 'n_detections'] = n_detections
        df_fiducials.at[j, 'x_mean'] = x_mean
        df_fiducials.at[j, 'x_sd'] = x_sd
        df_fiducials.at[j, 'x_fwhm'] = x_fwhm
        df_fiducials.at[j, 'y_mean'] = y_mean
        df_fiducials.at[j, 'y_fwhm'] = y_fwhm
        df_fiducials.at[j, 'y_sd'] = y_sd
        df_fiducials.at[j, 'z_mean'] = z_mean
        df_fiducials.at[j, 'z_sd'] = z_sd
        df_fiducials.at[j, 'z_fwhm'] = z_fwhm
        df_fiducials.at[j, 'n_images'] = n_images
        df_fiducials.at[j, 'detections_per_image'] = detections_per_image
        df_fiducials.at[j, 'photons_mean'] = photons_mean
        df_fiducials.at[j, 'photons_sd'] = photons_sd
        df_fiducials.at[j, 'x_deltaz_cor'] = x_deltaz_cor
        df_fiducials.at[j, 'y_deltaz_cor'] = y_deltaz_cor
        df_fiducials.at[j, 'z_deltaz_cor'] = z_deltaz_cor
        df_fiducials.at[j, 'x_z_step_cor'] = x_z_step_cor
        df_fiducials.at[j, 'y_z_step_cor'] = y_z_step_cor
        df_fiducials.at[j, 'z_z_step_cor'] = z_z_step_cor
        df_fiducials.at[j, 'x_deltaz_slope'] = x_deltaz_slope
        df_fiducials.at[j, 'y_deltaz_slope'] = y_deltaz_slope
        df_fiducials.at[j, 'z_deltaz_slope'] = z_deltaz_slope
        df_fiducials.at[j, 'x_madr'] = x_madr
        df_fiducials.at[j, 'y_madr'] = y_madr
        df_fiducials.at[j, 'z_madr'] = z_madr
        if num_time_points > 1:
            df_fiducials.at[j, 'time_point_separation'] = metrics_if[num_time_points-2, j] # largest time point separation

    df_fiducials['r_madr'] = np.sqrt(df_fiducials['x_madr']**2 + df_fiducials['y_madr']**2)
    df_fiducials['r_sd'] = np.sqrt(df_fiducials['x_sd']**2 + df_fiducials['y_sd']**2)

    return df_fiducials

def filter_fiducials(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Filter fiducials based on stats
    # Plot histograms of stats from n_detections to photons_sd
    logging.info('filter_fiducials')
    filter_cols = [
        "log_intensity",
        "n_detections",
        "x_sd",
        "y_sd",
        "z_sd",
        "photons_mean",
        "x_madr",
        "y_madr",
        "z_madr",
        "consensus_error",
        "time_point_separation"
    ]
    df_filt = pd.DataFrame({
        "colname": filter_cols,
        "lb": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ub": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    })

    # Small and sparse fiducials are likely noise
    detections_cutoff = config['min_fiducial_detections']
    area_cutoff = config['min_fiducial_size']
    quantile_outlier_cutoff = config['quantile_outlier_cutoff']
    quantile_max = 1 - quantile_outlier_cutoff
    quantile_min = quantile_outlier_cutoff
    sd_outlier_cutoff = config['sd_outlier_cutoff']

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

    # Set idx to True for all fiducials
    idx = np.ones(len(df_fiducials), dtype=bool)
    if sd_outlier_cutoff>0 or quantile_outlier_cutoff>0:
        # Get rid of those that are too far from the mean
        for j in range(len(df_filt)):
            col = df_filt.at[j, 'colname']
            lb = df_filt.at[j, 'lb']
            ub = df_filt.at[j, 'ub']
            if col in df_fiducials.columns:
                x = df_fiducials[col]
                if sd_outlier_cutoff > 0:
                    x_median = np.median(x)
                    x_sd = np.std(x)
                    x_mad = scipy.stats.median_abs_deviation(x, scale='normal')
                    # in the unlikely event that mad is fooled by a spike around the median, use sd
                    if x_mad == 0:
                        x_mad = x_sd
                    x_min = x_median - sd_outlier_cutoff * x_mad
                    x_max = x_median + sd_outlier_cutoff * x_mad
                else:
                    x_min = np.quantile(x,quantile_min)
                    x_max = np.quantile(x,quantile_max)
                if lb > 0:
                    idx = idx & (x >= x_min)
                    logging.info(f"Filtering {col}: {np.sum(x < x_min)} entries < {x_min:.2f} from total {len(df_fiducials)}")
                if ub > 0:
                    idx = idx & (x <= x_max)
                    logging.info(f"Filtering {col}: {np.sum(x > x_max)} entries > {x_max:.2f} from total {len(df_fiducials)}")
            else:
                logging.warning(f'Column {col} not found in df_fiducials')

    excluded_labels = pd.concat([excluded_labels, df_fiducials[idx == False]['label']])
    # Add to excluded labels the list in config['exclude_fiducials']
    excluded_labels = pd.concat([excluded_labels, config['excluded_fiducials']])
    # logging.info(f'Excluded labels: {excluded_labels}')
    logging.info(f'Filtering all: {np.sum(idx)} from total {len(df_fiducials)}')

    df_fiducials = df_fiducials[idx]
    df_fiducials = df_fiducials.reset_index(drop=True)

    if len(df_fiducials) == 0:
        logging.error('No fiducials left after filtering for stability and photons etc. Is outlier_cutoff too stringent?')
        return None, None
    hists_path = os.path.join(config['fiducial_dir'], 'histograms')
    for col in ['n_detections', 'n_images', 'detections_per_image', 'x_mean', 'x_sd', 'y_mean', 'y_sd', 'z_mean', 'z_sd', 'photons_mean', 'photons_sd', 'area', 'x_madr', 'y_madr', 'z_madr']:
        outpath = os.path.join(hists_path, f"hist_{col}")
        plot_histogram(df_fiducials[col], col, 'Fiducials', '', outpath, config)

    if config['debug']:
        outpath = os.path.join(config['fiducial_dir'], "x_sd_vs_y_sd")
        plot_scatter(df_fiducials['x_sd'], df_fiducials['y_sd'], 'x_sd (nm)', 'y_sd (nm)', 'x_sd vs y_sd', outpath, config)
        outpath = os.path.join(config['fiducial_dir'], "x_sd_vs_x_madr")
        plot_scatter(df_fiducials['x_sd'], df_fiducials['x_madr'], 'x_sd (nm)', 'x_madr (nm)', 'x_sd vs x_madr', outpath, config)
        outpath = os.path.join(config['fiducial_dir'], "y_sd_vs_y_madr")
        plot_scatter(df_fiducials['y_sd'], df_fiducials['y_madr'], 'y_sd (nm)', 'y_madr (nm)', 'y_sd vs y_madr', outpath, config)
        outpath = os.path.join(config['fiducial_dir'], "x_madr_vs_y_madr")
        plot_scatter(df_fiducials['x_madr'], df_fiducials['y_madr'], 'x_madr (nm)', 'y_madr (nm)', 'x_madr vs y_madr', outpath, config)

    # set excluded_labels from df.labels to 0 in df
    df.loc[df['label'].isin(excluded_labels.tolist()), 'label'] = 0
    return df, df_fiducials


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
            if np.sum(idx) > 0:
                plot_scatter(quantity[idx], dist[idx], xlabel, ylabel,
                             f'{quantity_name} vs distance', outpath, config)
    if config['debug']:
        # scatter plots of z for all pairs of fiducials
        for i in range(n_fiducials):
            for j in range(i+1, n_fiducials):
                xlabel = colnames[i]
                ylabel = colnames[j]
                idx = (z[:,i] != 0) & (z[:,j] != 0) & (~np.isnan(z[:,i])) & (~np.isnan(z[:,j]))
                if np.sum(idx) > 100:
                    z_diff = np.mean(z[idx,i] - z[idx,j])
                    z_diff_text = f'{int(z_diff):05d}'.replace('-', 'm')
                    filename = f"zdiff_{z_diff_text}_{ylabel}_vs_{xlabel}"
                    outpath = os.path.join(config['fiducial_dir'], "zscatter", filename)
                    logging.info(f'Plotting {outpath} with {np.sum(idx)} points')
                    plot_scatter(z[idx,i], z[idx,j], xlabel, ylabel,filename, outpath, config)

    # clustering of fiducials is of limited interest. Possibly this could be removed.
    if config['debug']:
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

def make_quality_metrics(df: pd.DataFrame, df_fiducials: pd.DataFrame, config: dict) ->  pd.DataFrame:
    # write csv file with quality metrics for fiducials. One row with columns variance of columns in df_fiducials
    logging.info('make_quality_metrics')
    df_metrics = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0]],
                              columns=['x_sd_mean', 'y_sd_mean', 'z_sd_mean', 'x_madr_mean', 'y_madr_mean', 'z_madr_mean', 'z_fiducial_sd', 'z_non_fiducial_sd'])
    df_metrics['x_sd_mean'] = df_fiducials['x_sd'].mean()
    df_metrics['y_sd_mean'] = df_fiducials['y_sd'].mean()
    df_metrics['z_sd_mean'] = df_fiducials['z_sd'].mean()
    df_metrics['fwhm_x_mean'] = df_fiducials['x_fwhm'].mean()
    df_metrics['fwhm_y_mean'] = df_fiducials['y_fwhm'].mean()
    df_metrics['fwhm_z_mean'] = df_fiducials['z_fwhm'].mean()
    df_metrics['x_deltaz_cor_mean'] = df_fiducials['x_deltaz_cor'].mean()
    df_metrics['y_deltaz_cor_mean'] = df_fiducials['y_deltaz_cor'].mean()
    df_metrics['z_deltaz_cor_mean'] = df_fiducials['z_deltaz_cor'].mean()
    df_metrics['x_madr_mean'] = df_fiducials['x_madr'].mean()
    df_metrics['y_madr_mean'] = df_fiducials['y_madr'].mean()
    df_metrics['z_madr_mean'] = df_fiducials['z_madr'].mean()
    df_metrics['z_fiducial_sd'] = df.loc[df['label']!=0,config['z_col']].std()
    df_metrics['z_non_fiducial_sd'] = df.loc[df['label']==0,config['z_col']].std()
    return df_metrics

def zstep_correct_fiducials(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return zstep_correct_fiducials_parallel(df_fiducials, df, config)

def zstep_correct_fiducials_parallel(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info('zstep_correct_fiducials_parallel')
    # If x1,... are taken then move them to x2,... first.
    # Check if backup column x_0 exists, if not then quit
    if not 'x_0' in df.columns:
        logging.error('No backup columns found in df')
        return df_fiducials, df
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    ndims = len(xyz_colnames)
    nfiducials = len(df_fiducials)
    fiducial_names = df_fiducials['name']
    fiducial_labels = df_fiducials['label']

    tasks = [(fiducial_labels[j], fiducial_names[j],
              df[df['label']==fiducial_labels[j]],df['label']==fiducial_labels[j], config) for j in range(nfiducials)]

    with multiprocessing.Pool(int(config['num_threads'])) as pool:
        results = pool.starmap(zstep_correct_fiducial, tasks)

    for df_cor, idx_cor in results:
        df.loc[idx_cor] = df_cor

    return df_fiducials, df


def zstep_correct_fiducial(fiducial_label: int, fiducial_name: str, df: pd.DataFrame, idx_cor: np.ndarray, config: dict) -> Tuple [pd.DataFrame, np.ndarray]:
    # Correct for z-step dependence. Assumes that impact is the same for all cycles
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    correct_z_only = 0
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
    dimnames = config['dimnames']
    ndims = len(xyz_colnames)
    sd_colnames = [config['x_sd_col'], config['y_sd_col'], config['z_sd_col']]
    # Create array to hold x,y,z for each cycle
    x_ct = np.zeros((total_cycles, frames_per_cycle), dtype=float)
    sd_ct = np.zeros((total_cycles, frames_per_cycle), dtype=float)
    x_ct.fill(np.nan)
    for k in range(ndims):
        # Skip x and y
        if correct_z_only and k<2:
            return df, idx_cor
        colname = xyz_colnames[k]
        logging.info(f'Correcting #{fiducial_label} label:{fiducial_name}:{colname}')
        # Get x,y,z values for each cycle
        for j in range(num_time_points):
            for i in range(num_cycles):
                idx = (
                    (df[config['cycle_col']] == i + min_cycle) &
                    (df[config['time_point_col']] == j + min_time_point)
                )
                df_sel = df[idx]
                cycle_index = i + j * num_cycles
                frame_index = df_sel[config['frame_col']] - min_frame + (df_sel[config['z_step_col']] - min_z_step) * num_frames
                x_ct[cycle_index, frame_index] = df_sel[colname].values
                sd_ct[cycle_index, frame_index] = df_sel[sd_colnames[k]].values
        # At this point x_ct and sd_ct contain the x,y,z values for each cycle/timepoint
        # x_ct has NAs for missing frames and sd_ct has zeros for missing frames
        dx_c = make_corrections_for_cycles(x_ct, sd_ct, config)
        c_z_step = make_corrections_for_zstep(x_ct, sd_ct, dx_c, config)
        x_ct_cor = apply_corrections_for_zstep(x_ct, c_z_step, config)
        sd_t = estimate_errors_for_zstep(x_ct_cor, config)
        # Transfer corrected values back to df
        for j in range(num_time_points):
            for i in range(num_cycles):
                idx = (
                        (df[config['cycle_col']] == i + min_cycle) &
                    (df[config['time_point_col']] == j + min_time_point)
                )
                cycle_index = i + j * num_cycles
                frame_index = df[idx][config['frame_col']] - min_frame + (df[idx][config['z_step_col']] - min_z_step) * num_frames
                df.loc[idx, colname] = x_ct_cor[cycle_index, frame_index]
                # Transfer corrected sd values back to df
                # For those elements without sd value from correction, use the old value
                # This can happen because there weren't enough detections to estimate the error
                non_nan_mask = ~np.isnan(sd_t[frame_index]) & (sd_t[frame_index] != 0)
                df.loc[idx, sd_colnames[k]] = np.where(non_nan_mask, sd_t[frame_index], df.loc[idx, sd_colnames[k]])

        dim = dimnames[k]
        outdir = os.path.join(config['fiducial_dir'], f"{fiducial_name}")
        outpath = os.path.join(outdir, f"{fiducial_name}_cor_{dim}_vs_frame")
        plot_scatter(df[config['image_id_col']], df[colname], 'image-ID', f'{dim} (nm)', f"{dim} corrected for z-step vs frame",
                     outpath, config)
        plotly_scatter(df[config['image_id_col']], df[colname], df[sd_colnames[k]], 'image-ID', f'{dim} (nm)', f"{dim} corrected for z-step vs frame",
                       outpath, config)
        # Plot fitted values on top of original values
        if config['plot_per_fiducial_fitting']:
            plot_fiduciual_zstep_fit(fiducial_label,df,dim, config)
    return df, idx_cor

def plot_fiduciual_zstep_fit(fiducial_index: int, df: pd.DataFrame,dim: str, config: dict) -> int:
    logging.info('plot_fiduciual_zstep_fit')
    # Plots the original and corrected values for a particular dim estimate of a fiducial
    outdir = os.path.join(config['output_dir'], "fiducial_zstep_fit")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"f_{fiducial_index}_d_{dim}_fit")
    x = df[config['image_id_col']]
    y_cor_col = config[f'{dim}_col']
    y_cor_sd_col = config[f'{dim}_sd_col']
    y_orig_col = f'{y_cor_col}_0'
    y_orig_sd_col = f'{y_cor_sd_col}_0'
    y_cor = df[y_cor_col]
    y_cor_sd = df[y_cor_sd_col]
    y_orig = df[y_orig_col]
    y_orig_sd = df[y_orig_sd_col]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    sc1 = ax1.scatter(x, y_orig, s=0.1, c=y_orig_sd, label='Original')
    # Add sd of y_orig to the plot
    sd_y = np.nanstd(y_orig)
    ax1.text(0.1, 0.9, f'sd = {sd_y:.2f}', transform=ax1.transAxes)
    ax1.set_xlabel('image-ID')
    ax1.set_ylabel(f"{dim} (nm)")
    ax1.set_title(f'Original {dim} fit for fid={fiducial_index}')
    ax1.legend()
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Original SD')

    sc2 = ax2.scatter(x, y_cor, s=0.1, c=y_cor_sd, label='Corrected')
    # Add sd of y_orig to the plot
    sd_y = np.nanstd(y_cor)
    ax2.text(0.1, 0.9, f'sd = {sd_y:.2f}', transform=ax2.transAxes)
    ax2.set_xlabel('image-ID')
    ax2.set_ylabel(f"{dim} (nm)")
    ax2.set_title(f'Corrected {dim} fit for fid={fiducial_index}')
    ax2.legend()
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Corrected SD')

    # Determine the combined y-range
    y_min = min(y_orig.min(), y_cor.min())
    y_max = max(y_orig.max(), y_cor.max())
    # Set the same y-range for both axes
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return 0

def make_corrections_for_cycles(x_ct: np.ndarray, sd_ct: np.ndarray, config: dict) -> np.ndarray:
    # Make corrections for cycles - add this number to each cycle of fiducial.
    # Meant to correct for drift during z-step compensation.
    # There may be nan's since not all fiducials have values at all frames
    # Fill in unknown values with zero
    logging.info('make_corrections_for_cycles')
    x_ct_masked = np.ma.masked_invalid(x_ct)
    sd_ct_masked = np.ma.masked_invalid(sd_ct)
    combined_mask = np.logical_or(x_ct_masked.mask, sd_ct_masked.mask)
    non_zero_mask = sd_ct != 0
    weights = np.full_like(sd_ct, np.nan, dtype=float)
    weights[non_zero_mask] = 1 / sd_ct[non_zero_mask]**2
    x_ct_masked_combined = np.ma.masked_array(x_ct, mask=combined_mask)
    weights_masked_combined = np.ma.masked_array(weights, mask=combined_mask)
    # Weighted average, ignoring NaNs
    c_cycle = -np.ma.average(x_ct_masked_combined, axis=1, weights=weights_masked_combined).filled(0)
    return c_cycle[:, None]

def make_corrections_for_zstep(x_ct: np.ndarray, sd_ct: np.ndarray, dx_c: np.ndarray, config: dict) -> np.ndarray:
    # Make corrections for zstep
    # There may be nan's since not all fiducials have values at all frames
    # Fill in unknown values with zeros
    # The correction is just the weighted average of the fiducials after correction for cycles (drift and offset)
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
        weights[non_zero_mask] = 1 / sd_ct_z[non_zero_mask]**2
        x_ct_masked_combined = np.ma.masked_array(x_ct_z, mask=combined_mask)
        weights_masked_combined = np.ma.masked_array(weights, mask=combined_mask)
        # Weighted average, ignoring NaNs
        if np.any(~x_ct_masked_combined.mask):
            c_z_step[i] = -np.ma.average(x_ct_masked_combined, weights=weights_masked_combined)
    return c_z_step

def apply_corrections_for_cycles(x_ct: np.ndarray, dx_c: np.ndarray, config: dict) -> np.ndarray:
    # Apply corrections for cycles
    logging.info('apply_corrections_for_cycles')
    x_ct_cor = x_ct + dx_c
    return x_ct_cor

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
    idx = valid_counts > 2 # Need at least 3 values to estimate sd and avoid tiny values
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

def drift_correct_detections(df: pd.DataFrame, df_fiducials: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info('drift_correct_detections')
    # Need at least 2 fiducials to make a consensus fit
    nfiducials = len(df_fiducials)
    if nfiducials < 6:
        logging.warning('Only have {nfiducials} fiducials. Less than 6 or so may not make a very good drift correction')

    noclobber = config['noclobber']
    x_col = ['x', 'y', 'z']
    xsd_col = ['x_sd', 'y_sd', 'z_sd']
    ndims = len(x_col)
    # if fiducials file does not exist, read them in
    outpath_x = os.path.join(config['cache_dir'], "fiducial_detections_x.npy")
    outpath_xsd = os.path.join(config['cache_dir'], "fiducial_detections_xsd.npy")
    if not os.path.exists(outpath_x) or not noclobber:
        x_ft, xsd_ft = extract_fiducial_detections(df, df_fiducials, config)
        if config['make_caches']:
            np.save(outpath_x, x_ft)
            np.save(outpath_xsd, xsd_ft)
    else:
        x_ft = np.load(outpath_x)
        xsd_ft = np.load(outpath_xsd)
        # Check that dimensions match the number of fiducials, if not throw error
        if x_ft.shape[1] != len(df_fiducials):
            logging.error(f'Number of fiducials in df_fiducials ({len(df_fiducials)}) does not match the number of fiducials in x_ft ({x_ft.shape[1]})')
            raise RuntimeError('Mismatch with cached fiducials. Delete cache or set noclobber=0.')


    # fit fiducials, interpolate across all time points and give uncertainties to interpolated areas
    outpath_x = os.path.join(config['cache_dir'], "fiducial_fits_x.npy")
    outpath_xsd = os.path.join(config['cache_dir'], "fiducial_fits_xsd.npy")
    if not os.path.exists(outpath_x) or not noclobber:
        x_fit_ft, xsd_fit_ft = fit_fiducial_detections( x_ft, xsd_ft, config)
        if config['make_caches']:
            np.save(outpath_x, x_fit_ft)
            np.save(outpath_xsd, xsd_fit_ft)
    else:
        x_fit_ft = np.load(outpath_x)
        xsd_fit_ft = np.load(outpath_xsd)
        # Check that dimensions match the number of fiducials
        if x_fit_ft.shape[1] != len(df_fiducials):
            logging.error(f'Number of fiducials in df_fiducials ({len(df_fiducials)}) does not match the number of fiducials in x_fit_ft ({x_fit_ft.shape[1]})')
            raise RuntimeError('Mismatch with cached fiducials. Delete cache or set noclobber=0.')
    # group fiducials to be zero centred
    if nfiducials >= 2:
        x_fit_ft, xsd_fit_ft = group_fiducial_fits(x_fit_ft, xsd_fit_ft, config)
    else:
        logging.warning('Not enough fiducials to make a consensus drift correction. Just using the one.')

    # Fit drift correction to zero-centred fiducials - including across time-point boundaries
    x_t, x_err, err_f = make_drift_corrections(df_fiducials, x_fit_ft, xsd_fit_ft, config)
    # add err_f and fitting error to df_fiducials to enable examination of the error in the consensus later
    df_fiducials['consensus_error'] = err_f
    df_fiducials['fitting_error'] = np.nanmean(xsd_fit_ft, axis=(0,2)) # average over dims and time
    if config['plot_per_fiducial_fitting']:
        plot_fitted_fiducials(df_fiducials, x_fit_ft, xsd_fit_ft, config)

    # Save and plot drift correction with error bars
    df_drift = pd.DataFrame({
        config['image_id_col']: np.arange(x_t.shape[1]),
        'x': x_t[0,:], 'x_sd': x_err[0,:], 'y': x_t[1,:], 'y_sd': x_err[1,:], 'z': x_t[2,:], 'z_sd': x_err[2,:]
    })
    # Plot the drift correction with error bars
    plot_drift_correction(df_drift, config)
    # correct detections
    df = apply_corrections(df, x_t, config)
    # Make derivatives for checking correction
    # This doesn't really make sense unless you can do it on noise-filtered fiducials
    # df_drift = compute_time_derivates(df, df_drift, config)
    # plot_time_derivatives(df_drift, config)
    output_path = os.path.join(config['output_dir'], "drift_correction.tsv")
    df_drift.to_csv(output_path, sep='\t', index=False)
    return df, df_fiducials

def plot_fitted_fiducials(df_fiducials: pd.DataFrame, x_fit_ft: np.ndarray, xsd_fit_ft: np.ndarray, config: dict) -> int:
    if config['multiprocessing']:
        return plot_fitted_fiducials_parallel(df_fiducials, x_fit_ft, xsd_fit_ft, config)
    else:
        logging.info('plot_fitted_fiducials')
        ndims = x_fit_ft.shape[0]
        nfiducials = x_fit_ft.shape[1]
        for j in range(nfiducials):
            for k in range(ndims):
                fiducial_name = df_fiducials.name[j]
                plot_fitted_fiducial(fiducial_name, j, k, x_fit_ft, xsd_fit_ft, config)
    return 0

def plot_fitted_fiducial(fiducial_name, j, k, x_fit_ft, xsd_fit_ft, config):
    outdir = os.path.join(config['fiducial_dir'], f"{fiducial_name}")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{fiducial_name}_{config['dimnames'][k]}_fit_vs_frame")
    plotly_scatter(np.arange(x_fit_ft.shape[2]), x_fit_ft[k, j, :], None,
                   'image-ID', f'{config["dimnames"][k]} (nm)', f'{config["dimnames"][k]} fit vs frame', outpath, config)

def plot_fitted_fiducials_parallel(df_fiducials: pd.DataFrame, x_fit_ft: np.ndarray, xsd_fit_ft: np.ndarray, config: dict) -> int:
    logging.info('plot_fitted_fiducials_parallel')

    ndims = x_fit_ft.shape[0]
    nfiducials = x_fit_ft.shape[1]
    tasks = [(df_fiducials.name[j], j, k, x_fit_ft, xsd_fit_ft, config) for j in range(nfiducials) for k in range(ndims)]

    with multiprocessing.Pool(int(config['num_threads'])) as pool:
        pool.starmap(plot_fitted_fiducial, tasks)

    return 0

def combine_fiducial_fits(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) ->  Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info('combine_fiducial_fits')
    # Combine the fits for all fiducials
    # method = "weighted_mean" # "median" or "weighted_mean"
    method = config['consensus_method']
    w = np.zeros_like(x_ft)
    xsd_f = np.sqrt(np.sum(xsd_ft*xsd_ft, axis=2))
    w[:] = 1 / xsd_f[:, :, np.newaxis]**2
    if method == "median":
        x_t = np.median(x_ft, axis=1)
    elif method == "weighted_mean":
        x_t = np.average(x_ft, axis=1, weights=w)
    else:
        logging.error(f'Unknown method {method} for combining fiducials')
        return x_ft, xsd_ft, xsd_f
    # err per time point is a dim x times array
    x_err = np.sqrt(np.average((x_ft - x_t[:, None])**2, axis=1, weights=w))
    # err per fiducial is a 1D array of length n_fiducials - dim and time summed over
    f_err = np.sqrt(np.average((x_ft - x_t[:, None])**2, axis=(0,2), weights=w))
    return x_t, x_err, f_err

def make_drift_corrections(df_fiducials: pd.DataFrame, x_fit_ft: np.ndarray, xsd_fit_ft: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info('make_corrections')
    ndims = x_fit_ft.shape[0]
    nfiducials = x_fit_ft.shape[1]
    x_col = ['x', 'y', 'z']
    # Take all the fiducial fits and combine them into the drift estimate
    x_t, xsd_t, err_f = combine_fiducial_fits(x_fit_ft, xsd_fit_ft, config)
    # Plot the combined fit with the individual fits
    # TODO: parallelise this plotting loop
    for k in range(ndims):
        logging.info(f'Plotting combined fitted corrections for {x_col[k]}')
        outdir = config['output_dir']
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f"combined_corrections_{x_col[k]}_vs_frame")
        plt.figure(figsize=(10, 6))
        for j in range(nfiducials):
            label = df_fiducials.label[j]
            plt.scatter(np.arange(x_fit_ft.shape[2]), x_fit_ft[k,j,:], s=0.5, label=f'{label}')
        plt.scatter(np.arange(x_fit_ft.shape[2]), x_t[k,:], s=0.5, c='black', label='fit')
        plt.legend(markerscale=4, handletextpad=0.1, loc='best', fancybox=True, framealpha=1, fontsize='medium')
        plt.xlabel('image-ID')
        plt.ylabel(f"{x_col[k]} (nm)")
        plt.title(f'Fits for {x_col[k]}')
        plt.savefig(outpath)
        plt.close()
    # Make the corrections start at zero
    x_t = x_t - x_t[:, [0]]
    return x_t, xsd_t, err_f


def minimize_fiducial_fit_variance(x_ft: np.ndarray, xsd_ft: np.ndarray,config: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Minimise the variance of the fiducial fits at each time point
    # One of the options for grouping together fiducial fits
    os_name = platform.system()
    # multiprocessing does not work with scipy.optimize.minimize() on Windows
    if os_name != "Windows" and config['multiprocessing']:
       return minimize_fiducial_fit_variance_parallel(x_ft, xsd_ft,config)
    logging.info('minimize_fiducial_fit_variance')
    # Finds the offsets that minimise the variance of the fiducial fits at each time point
    optimise_method = "L-BFGS-B" # L-BFGS-B, Powell, CG, BFGS, Nelder-Mead, TNC, COBYLA, SLSQP, Newton-CG, trust-ncg, trust-krylov, trust-exact, trust-constr
    dimensions = ['x', 'y', 'z']
    nfiducials = x_ft.shape[1]
    ndimensions = x_ft.shape[0]
    x_ret = x_ft.copy()
    w = 1 / xsd_ft**2 # weight to be used in the cost function

    def apply_offsets(offsets, x):
        nfiducials = x.shape[0]
        x_shifted = x.copy()
        # Apply offset to all fiducials except the first
        x_shifted[1:nfiducials, :] += offsets[:nfiducials-1, np.newaxis]
        if np.any(np.isnan(x_shifted)):
            logging.error('NaN in x_shifted')
        return x_shifted
    def make_bounds(x):
        # Make bounds for the offsets
        # Bound is twice the distance from the first fiducial in either direction (overly conservative but better than nothing)
        nfiducials = x.shape[0]
        bounds = []
        for i in range(nfiducials - 1):
            d = np.abs(np.mean(x[i+1,:] - x[0,:]))
            bounds.append((-2*d, 2*d))
        return bounds
    def variance_cost(offsets, x, weight_t):
        x_shifted = apply_offsets(offsets, x)
        # At each time point, calculate the variance of the fiducial fits across time points
        # Weight this by the uncertainties at each time point
        var_t = np.nanvar(x_shifted, axis=0)
        ret = np.sum(var_t * weight_t)
        # Make Jacobian
        nfiducials = x.shape[0]
        x_shifted_mean_t = np.nanmean(x_shifted, axis=0)
        jac = (2/nfiducials) * np.sum((x_shifted - x_shifted_mean_t) * weight_t, axis=1)
        # Remove first element of jac, to match offsets
        jac = jac[1:]
        debug = False
        if debug:
            logging.info(f'Cost: {ret} offsets: {offsets} jac: {jac}')
            logging.info(f'Jacobian: {jac}')
        # combine the cost and jacobian into a tuple
        ret = (ret, jac)
        return ret

    for j in range(ndimensions):
        logging.info(f'Grouping fiducials fits to {dimensions[j]}')
        weight_t = np.sum(w[j,:,:], axis=0) / np.sum(w[j,:,:])  # make weight sum to 1 and f(t)

        bounds = make_bounds(x_ret[j,:,:])
        initial_offsets = np.zeros(nfiducials - 1)
        initial_cost = variance_cost(initial_offsets, x_ret[j,:,:], weight_t)
        result = scipy.optimize.minimize(variance_cost, initial_offsets, args=(x_ret[j,:,:], weight_t),
                                        method=optimise_method, jac=True, bounds=bounds, options = {'maxiter': 1e5, 'disp': True})
        optimal_offsets = result.x
        final_cost = result.fun
        logging.info(f'Initial cost: {initial_cost[0]} Final cost: {final_cost}')
        x_ret[j,:,:] = apply_offsets(optimal_offsets, x_ret[j,:,:])
    return x_ret, xsd_ft

def group_fiducial_fits(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Group fiducials before taking an average over them
    # Optimisation seems to be reliable. Perhaps not faster than round_robin though.
    # Zero is a simple method that just subtracts the average of each fiducial from all time points
    # This forces them to be zero centred but doesn't group them as closely.
    group_by = "optimise" # "round_robin" or "zero" or "optimise"
    if group_by == "optimise":
        x_ret, xsd_ret = minimize_fiducial_fit_variance(x_ft, xsd_ft,config)
    elif group_by == "round_robin":
        x_ret, xsd_ret = group_fiducial_fits_round_robin(x_ft, xsd_ft, config)
    elif group_by == "zero":
        x_ret, xsd_ret = group_fiducial_fits_to_zero(x_ft, xsd_ft, config)
    return x_ret, xsd_ret

def group_fiducial_fits_to_zero(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('group_fiducial_fits_to_zero')
    ndimensions = x_ft.shape[0]
    nfiducials = x_ft.shape[1]
    x_ret = x_ft.copy()
    for i in range(nfiducials):
        for j in range(ndimensions):
            w = 1 / xsd_ft**2
            weighted_average = np.average(x_ret[j,i,:], weights=w[j,i,:])
            x_ret[j,i,:] = x_ret[j,i,:] - weighted_average
    return x_ret, xsd_ft

def group_fiducial_fits_round_robin(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('group_fiducial_fits_round_robin')
    # Group fiducials to be zero centred
    ndimensions = x_ft.shape[0]
    nfiducials = x_ft.shape[1]
    w = 1 / xsd_ft**2
    # group to zero first
    x_ret, xsd_ft = group_fiducial_fits_to_zero(x_ft, xsd_ft, config)
    # Try and group them together closer with a second pass of "round robin" squeezing
    # This is a bit of a hack, but it seems to work.
    for k in range(np.min((nfiducials,10))):
        for i in range(nfiducials):
            for j in range(ndimensions):
                # find the offset that minimises the RMS distance
                not_this_fiducial_index = np.arange(nfiducials) != i
                offset = np.average(x_ret[j, not_this_fiducial_index, :] - x_ret[j, i, :], weights=w[j, not_this_fiducial_index, :])
                x_ret[j, i, :] = x_ret[j, i, :] + offset
                logging.info(f'pass: {k} fiducial: {i} dimension: {j}  offset: {offset}')
    return x_ret, xsd_ft


def fit_fiducial_step_parallel(i, k, fitting_intervals, x_ft, xsd_ft, config):
    # Assumes that x_ft and xsd_ft are 1D arrays with indexing by dim and fiducial already done
    logging.info(f"fit_fiducial_step_parallel: fid_idx: {i} dim_idx: {k}")
    x_fit_ft = np.zeros_like(x_ft)
    x_fit_ft.fill(np.nan)
    xsd_fit_ft = np.zeros_like(xsd_ft)
    xsd_fit_ft.fill(np.nan)

    for j in range(len(fitting_intervals) - 1):
        idx = np.arange(fitting_intervals[j], fitting_intervals[j + 1])
        logging.info(f'fit_fiducial_step: fid_idx: {i} dim_idx: {k} seg_idx: {j} ')
        x_fit_ft[idx], xsd_fit_ft[idx] = fit_fiducial_step(x_ft[idx], xsd_ft[idx], config)
        y = x_ft[idx]
        ysd = xsd_ft[idx]
        y_fit = x_fit_ft[idx]
        if np.sum(~np.isnan(y)) == 0 or np.sum(~np.isnan(ysd)) == 0 or np.sum(~np.isnan(y_fit)) == 0:
            logging.warning(
                f'No valid data for fitting in fit_fiducial_detections() for fiducial id {i+1} dimension {k} interval {j}')
            if j > 0:
                x_fit_ft[idx] = x_fit_ft[fitting_intervals[j] - 1]
                xsd_fit_ft[idx] = xsd_fit_ft[fitting_intervals[j] - 1]
            else:
                x_fit_ft[idx] = 0
                xsd_fit_ft[idx] = 0
                continue
        elif config['plot_per_fiducial_fitting']:
            plot_fiduciual_step_fit(i, j, k, x_ft[idx], xsd_ft[idx], x_fit_ft[idx],xsd_fit_ft[idx], config)

    for j in range(len(fitting_intervals) - 2, 0, -1):
        idx = np.arange(fitting_intervals[j - 1], fitting_intervals[j])
        if np.sum(xsd_fit_ft[idx] == 0) > 0:
            # If there are no valid data points in the interval, use the previous interval
            logging.info(f"No valid data in fit_fiducial_detections() for fiducial id {i+1}")
            xsd_fit_ft[idx] = xsd_fit_ft[fitting_intervals[j]]
            x_fit_ft[idx] = x_fit_ft[fitting_intervals[j]]

    return i, k, x_fit_ft, xsd_fit_ft

def plot_fiduciual_step_fit(fiducial_index: int, interval_index: int, dimension_index: int, y: np.ndarray, ysd: np.ndarray, y_fit: np.ndarray, ysd_fit: np.ndarray, config: dict) -> int:
    logging.info('plot_fiduciual_step_fit')
    dimnames = config['dimnames']
    dim = dimnames[dimension_index]
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
    polynomial_degree = config['polynomial_degree']
    use_weights_in_fit = (config['use_weights_in_fit']!=0)
    extrapolate_to_end = True
    median_filter_size = 100
    outlier_threshold = 3
    w = np.full_like(xt_sd, np.nan)
    non_nan_indices = ~np.isnan(xt) & (xt_sd != 0)
    nan_indices = ~non_nan_indices
    w[non_nan_indices] = 1 / xt_sd[non_nan_indices] # avoid division by zero

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

def fit_fiducial_detections(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    return fit_fiducial_detections_parallel(x_ft, xsd_ft, config)

def fit_fiducial_detections_parallel(x_ft: np.ndarray, xsd_ft: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    logging.info('fit_fiducial_detections_parallel')
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
    frames_per_z_step = num_frames

    if config['fitting_interval'] == 'time_point':
        fitting_intervals = np.arange(0, total_frames + frames_per_time_point, frames_per_time_point)
    elif config['fitting_interval'] == 'cycle':
        fitting_intervals = np.arange(0, total_frames + frames_per_cycle, frames_per_cycle)
    elif config['fitting_interval'] == 'z_step':
        fitting_intervals = np.arange(0, total_frames + frames_per_z_step, frames_per_z_step)
    else:
        logging.error(f"Unknown fitting interval: {config['fitting_interval']}")

    x_fit_ft = np.zeros_like(x_ft)
    x_fit_ft.fill(np.nan)
    xsd_fit_ft = np.zeros_like(xsd_ft)
    xsd_fit_ft.fill(np.nan)

    with multiprocessing.Pool(int(config['num_threads'])) as pool:
        results = pool.starmap(fit_fiducial_step_parallel,
                               [(i, k, fitting_intervals, x_ft[k,i,:], xsd_ft[k,i,:], config) for i in range(nfiducials) for k in
                                range(ndim)])

    for i, k, x_fit, xsd_fit in results:
        x_fit_ft[k, i, :] = x_fit
        xsd_fit_ft[k, i, :] = xsd_fit

    if np.sum(xsd_fit_ft == 0) > 0:
        logging.error('Zeros in xsd_fit_ft')
    if np.sum(np.isnan(x_fit_ft)) > 0:
        logging.error('NaN in x_fit_ft')
    return x_fit_ft, xsd_fit_ft


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

