import logging
import plotly.express as px
import pandas as pd
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile
import multiprocessing
from PIL import Image, ImageDraw, ImageFont
from zedtool.srxstats import z_means_by_marker, z_max_by_marker
from zedtool.detections import fwhm_from_points
from zedtool.image import add_axes_and_scale_bar

# Prints some debugging plots for an SRX dataset.
# Takes a corrected and an uncorrected table of detections, registers the rows and finds the corrections.
# Write out a table with both corrected and uncorrected z.

def construct_plot_path(filename: str, filetype: str, config: dict) -> str:
    """
    Constructs the full path for saving a plot file, ensuring the output directory exists.

    Parameters
    ----------
    filename : str
        Base filename for the plot.
    filetype : str
        File extension/type for the plot (e.g., 'png', 'html').
    config : dict
        Configuration dictionary containing output directory info.

    Returns
    -------
    str
        Full path to the plot file.
    """
    use_plots_dir = False
    # if filename doesn't contain config['output_dir'], prepend it
    if not filename.startswith(config['output_dir']):
        figure_path = os.path.join(config['output_dir'], f"{filename}.{filetype}")
    else:
        figure_path = f"{filename}.{filetype}"
    # Ensure the last directory is "plots"
    if use_plots_dir:
        dir_path, file_name = os.path.split(figure_path)
        if os.path.basename(dir_path) != "plots":
            figure_path = os.path.join(dir_path, "plots", file_name)
    # If figure_dir doesn't exist, create it
    figure_dir = os.path.dirname(figure_path)
    os.makedirs(figure_dir, exist_ok=True)
    return figure_path

def plot_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, filename: str, config: dict) -> int:
    """
    Plots a scatter plot of x vs y and saves it as a PNG file.

    Parameters
    ----------
    x : np.ndarray
        Array of x values.
    y : np.ndarray
        Array of y values.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    filename : str
        Filename for saving the plot.
    config : dict
        Configuration dictionary for output settings.

    Returns
    -------
    int
        0 on success.
    """
    filetype = "png"
    plt.figure()
    # Scale point size with number of detections
    if len(x) > 0:
        point_size = 100 / len(x)
        point_size = np.max([point_size, 0.01])
        point_size = np.min([point_size, 1.0])
    else:
        point_size = 1.0
        logging.warning(f"Empty x array in plot_scatter: {filename}")
    plt.scatter(x, y, s=point_size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    figure_path = construct_plot_path(filename, filetype, config)
    plt.savefig(figure_path, dpi=300)
    plt.close()
    return 0

def plotly_scatter(x: np.ndarray, y: np.ndarray, y_err: np.ndarray, xlabel: str, ylabel: str, title: str, filename: str, config: dict) -> int:
    """
    Plots an interactive scatter plot with error bars using Plotly and saves as HTML.

    Parameters
    ----------
    x : np.ndarray
        Array of x values.
    y : np.ndarray
        Array of y values.
    y_err : np.ndarray
        Array of y error values.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    filename : str
        Filename for saving the plot.
    config : dict
        Configuration dictionary for output settings.

    Returns
    -------
    int
        0 on success.
    """
    filetype = "html"
    figure = px.scatter(x=x, y=y, error_y=y_err, title=title, labels={xlabel: xlabel, ylabel: ylabel})
    # if filename doesn't contain config['output_dir'], prepend it
    figure_path = construct_plot_path(filename, filetype, config)
    figure.write_html(figure_path)
    return 0

def plot_histogram(x: np.ndarray, xlabel: str, ylabel: str, title: str, filename: str, config: dict) -> int:
    """
    Plots a histogram of x and saves it as a PNG file.

    Parameters
    ----------
    x : np.ndarray
        Array of values to plot.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    filename : str
        Filename for saving the plot.
    config : dict
        Configuration dictionary for output settings.

    Returns
    -------
    int
        0 on success.
    """
    hist_bins = 100 # TODO: make this a config option
    filetype = "png"
    plt.figure()
    plt.hist(x, bins=hist_bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if "log" in ylabel.lower():
        plt.yscale("log")
    figure_path = construct_plot_path(filename, filetype, config)
    plt.savefig(figure_path, dpi=300)
    plt.close()
    return 0

def plot_detections(df: pd.DataFrame, filename: str, config: dict):
    """
    Plots projections and histograms of detection coordinates from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing detection data.
    filename : str
        Filename for saving the plot.
    config : dict
        Configuration dictionary for column names and output settings.

    Returns
    -------
    None
        Saves the plot to file.
    """
    logging.info("plot_detections")
    # plot projections  of det_xyz
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # scatter plot of x,y in top left
    sc = ax[0, 0].scatter(df[config['x_col']], df[config['y_col']], s=0.01,alpha=0.05, c=df[config['z_step_col']])
    ax[0, 0].set_xlabel('x (nm)')
    ax[0, 0].set_ylabel('y (nm)')
    cbar = plt.colorbar(sc, ax=ax[0, 0])
    cbar.set_label('z-step')
    # scatter plot of x,z in bottom left
    ax[1, 0].scatter(df[config['x_col']], df[config['z_col']], s=0.01,alpha=0.05, c=df[config['z_step_col']])
    ax[1, 0].set_xlabel('x (nm)')
    ax[1, 0].set_ylabel('z (nm)')
    # scatter plot of y,z in top right
    ax[0, 1].scatter(df[config['y_col']], df[config['z_col']], s=0.01,alpha=0.05, c=df[config['z_step_col']])
    ax[0, 1].set_xlabel('y (nm)')
    ax[0, 1].set_ylabel('z (nm)')
    # histogram of z in bottom right
    ax[1, 1].hist(df[config['z_col']], bins=100)
    ax[1, 1].set_xlabel('z (nm)')
    ax[1, 1].set_ylabel('count')
    figure_path = construct_plot_path(filename, "png", config)
    plt.savefig(figure_path, dpi=600)
    plt.close()

def plot_binned_detections_stats(n_xy: np.ndarray,mean_xy: np.ndarray, sd_xy: np.ndarray, filename: str, config: dict):
    """
    Plots binned detection statistics including log counts, mean, and standard deviation.

    Parameters
    ----------
    n_xy : np.ndarray
        2D array of detection counts per bin.
    mean_xy : np.ndarray
        2D array of mean values per bin.
    sd_xy : np.ndarray
        2D array of standard deviation values per bin.
    filename : str
        Filename for saving the plot.
    config : dict
        Configuration dictionary for output settings.

    Returns
    -------
    None
        Saves the plot to file.
    """
    logging.info("plot_binned_detections_stats")
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)
    im=ax[0, 0].imshow(np.log10(1+n_xy).T,origin='lower')
    ax[0, 0].set_axis_off()
    plt.colorbar(im, ax=ax[0,0], label='log10(n)')
    im=ax[0, 1].imshow(mean_xy.T,origin='lower')
    ax[0, 1].set_axis_off()
    plt.colorbar(im, ax=ax[0,1], label='nm')
    im=ax[1, 0].imshow(sd_xy.T,origin='lower')
    ax[1, 0].set_axis_off()
    plt.colorbar(im, ax=ax[1,0], label='nm')
    ax[1, 1].scatter(np.log10(1 + n_xy).flatten(), sd_xy.flatten(), s=0.1, c=mean_xy.flatten())
    plt.colorbar(im, ax=ax[1, 1], label='mean(z) nm')
    ax[2, 0].scatter(sd_xy.flatten(), mean_xy.flatten(), s=0.1, c=np.log10(1+n_xy).flatten())
    plt.colorbar(im, ax=ax[2, 0], label='log10(n)')
    ax[2, 1].scatter(np.log10(1+n_xy).flatten(), mean_xy.flatten(),s=0.1, c=sd_xy.flatten())
    plt.colorbar(im, ax=ax[2, 1], label='sd(z) nm')
    ax[0, 0].set_title('log_10(n)')
    ax[0, 1].set_title('mean(z) (nm)')
    ax[1, 0].set_title('sd(z) (nm)')
    ax[1, 1].set_xlabel('log_10(n)')
    ax[1, 1].set_ylabel('sd(z) nm')
    ax[2, 0].set_xlabel('sd(z) nm')
    ax[2, 0].set_ylabel('mean(z) nm')
    ax[2, 1].set_xlabel('log_10(n)')
    ax[2, 1].set_ylabel('mean(z) nm')
    figure_path = construct_plot_path(filename, "png", config)
    fig.savefig(figure_path, dpi=600)
    plt.close()

def plot_summary_stats(df: pd.DataFrame, det_xyz: np.ndarray, config: dict):
    """
    Plots summary statistics for detections and other quantities.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing detection data.
    det_xyz : np.ndarray
        Array of detection coordinates.
    config : dict
        Configuration dictionary for column names and output settings.

    Returns
    -------
    None
        Saves the plot(s) to file.
    """
    plot_histogram(df[config['x_sd_col']], f"{config['x_sd_col']} (nm)", 'Detections',
                   f"Detections {config['x_sd_col']}", os.path.join('summary_plots', 'x_sd_histogram'), config)
    plot_histogram(df[config['y_sd_col']], f"{config['y_sd_col']} (nm)", 'Detections',
                   f"Detections {config['y_sd_col']}", os.path.join('summary_plots','y_sd_histogram'), config)
    plot_histogram(df[config['z_sd_col']], f"{config['z_sd_col']} (nm)", 'Detections',
                   f"Detections {config['z_sd_col']}", os.path.join('summary_plots','z_sd_histogram'), config)
    plot_histogram(df[config['z_step_col']], 'z-step', 'Detections',
                   "Detections by z-step", os.path.join('summary_plots','zstep_histogram') , config)
    plot_histogram(df[config['deltaz_col']], f"{config['deltaz_col']} (nm)", 'Detections',
                   'Detections delta z', os.path.join('summary_plots','delta_z_histogram'), config)
    plot_scatter(df[config['image_id_col']], df[config['z_col']], f"{config['image_id_col']}", f"{config['z_col']} (nm)",
                 f"{config['z_col']} vs {config['image_id_col']}", os.path.join('summary_plots','z_vs_t'), config)
    plotly_scatter(df[config['image_id_col']], df[config['z_col']],None, f"{config['image_id_col']}", f"{config['z_col']} (nm)",
                   f"{config['z_col']} vs {config['image_id_col']}", os.path.join('summary_plots','z_vs_t'), config)
    plot_scatter(df[config['image_id_col']], df[config['photons_col']], f"{config['image_id_col']}", f"{config['photons_col']}",
                 f"{config['photons_col']} vs {config['image_id_col']}",os.path.join('summary_plots','photon_count_vs_t'), config)
    plotly_scatter(df[config['image_id_col']], df[config['photons_col']], None, f"{config['image_id_col']}", f"{config['photons_col']}",
                   f"{config['photons_col']} vs {config['image_id_col']}",os.path.join('summary_plots','photon_count_vs_t'), config)
    plot_scatter(df[config['image_id_col']], df[config['z_step_col']], f"{config['image_id_col']}", f"{config['z_step_col']}",
                 f"{config['z_step_col']} vs {config['image_id_col']}",os.path.join('summary_plots','zstep_vs_t'), config)
    plot_scatter(df[config['deltaz_col']] , df[config['z_col']],
                 f"{config['deltaz_col']} (nm)", f"{config['z_col']} (nm)",
                 f"{config['z_col']} vs {config['deltaz_col']}", os.path.join('summary_plots','z_vs_delta_z'), config)

    for colname in [config['image_id_col'], config['z_step_col'], config['cycle_col'], config['time_point_col']]:
        z_mean, t = z_means_by_marker(det_xyz, df[colname].values)
        plot_scatter(t, z_mean, colname, f'mean(z) per {colname} (nm)',
                     f"mean(z) vs {colname}", os.path.join('summary_plots',f'z_mean_per_{colname}_vs_{colname}'), config)
    # For zstep, save as a tsv file with diff(z_mean)
    # This file can be useful in determining z_step_step empirically if it's not known
    # Alternatively, check the logfile for the estimate therein.
    z_mean, z_step = z_means_by_marker(det_xyz, df[config['z_step_col']].values)
    z_max, z_step = z_max_by_marker(det_xyz, df[config['z_step_col']].values)
    df_z = pd.DataFrame({'z_step': z_step, 'z_mean': z_mean, 'z_max': z_max})
    df_z['diff_z_mean'] = df_z['z_mean'].diff()
    df_z['diff_z_max'] = df_z['z_max'].diff()
    df_z.to_csv(os.path.join(config['output_dir'], os.path.join('summary_plots','z_mean_max_vs_z_step.tsv')), sep='\t', index=False)

def stats_text(x: np.ndarray,title: str) -> str:
    """
    Calculates summary statistics (count, mean, standard deviation, FWHM) for an array and returns as formatted text.

    Parameters
    ----------
    x : np.ndarray
        Array of values to summarize.
    title : str
        Title for the statistics summary.

    Returns
    -------
    str
        Formatted statistics summary text.
    """
    # Calculate some statistics
    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x)
    fwhm_x = fwhm_from_points(x)
    text = f"{title}\n"
    text += f"n = {n}\n"
    text += f"mean = {mean_x:.2f}\n"
    text += f"sd = {sd_x:.2f}\n"
    text += f"fwhm = {fwhm_x:.2f}\n"
    return text

def plot_fiducial_rois(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> int:
    """
    Plots fiducial regions of interest (ROIs) on the detections image and saves annotated images.

    Parameters
    ----------
    df_fiducials : pd.DataFrame
        DataFrame containing fiducial ROI coordinates and labels.
    df : pd.DataFrame
        DataFrame containing detection data.
    config : dict
        Configuration dictionary for output settings and image resolution.

    Returns
    -------
    int
        0 on success.
    """
    # Plot the fiducials on the detections image so that they can be identified in the image
    logging.info("plot_fiducial_rois")
    detections_img_file = 'binned_detections_2d.tif'
    fiducials_plot_file = 'fiducials_plot'
    line_intensity = 228
    # read in the image and the segmentation from tifs
    img_filt = tifffile.imread(os.path.join(config['output_dir'], detections_img_file))
    # scale the image to 0-255
    im = 255 * (np.log10(img_filt+1)/np.log10(np.max(img_filt)+1))
    im = im.astype(np.uint8)
    max_x = np.max(df_fiducials['max_x'])
    max_y = np.max(df_fiducials['max_y'])
    # If the image is smaller than the fiducials, pad it. This can happen if, after correction,
    # the fiducials get smaller and the image is cropped to the fiducials.
    if max_x > im.shape[1] or max_y > im.shape[0]:
        logging.warning(f"Image size {im.shape} is smaller than fiducials (max_y,max_x) = ({max_y},{max_x}). Padding image.")
        # Pad the image with zeros
        pad_x = max(0, max_x - im.shape[1])
        pad_y = max(0, max_y - im.shape[0])
        im = np.pad(im, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
    # loop over rows of df
    for j in range(len(df_fiducials)):
        im[int(df_fiducials.min_y[j]):int(df_fiducials.max_y[j]-1), int(df_fiducials.min_x[j])] = line_intensity
        im[int(df_fiducials.min_y[j]):int(df_fiducials.max_y[j]-1), int(df_fiducials.max_x[j]-1)] = line_intensity
        im[int(df_fiducials.min_y[j]), int(df_fiducials.min_x[j]):int(df_fiducials.max_x[j]-1)] = line_intensity
        im[int(df_fiducials.max_y[j]-1), int(df_fiducials.min_x[j]):int(df_fiducials.max_x[j]-1)] = line_intensity
        # This line provides a shield from cropping of columns of pixels between characters in the label
        if int(df_fiducials.max_x[j] + 40) < im.shape[1]:
            im[int(df_fiducials.max_y[j]-1), int(df_fiducials.max_x[j]-1):int(df_fiducials.max_x[j] + 40)] = line_intensity
    imp = Image.fromarray(im)
    draw = ImageDraw.Draw(imp)
    for fontname in ["DejaVuSansMono", "couri", "Courier New", "arial", "LiberationSans-Regular"]:
        try:
            font = ImageFont.truetype(fontname, 10)
            break
        except OSError:
            pass
    # Add text, The underscores are to protect the text from being cut up by the row/col removal below
    for j in range(len(df_fiducials)):
        # underline_str = "_" * 6
        # label = f"{underline_str}\nf_{df_fiducials.label[j]:04d}"
        label = f"f_{df_fiducials.label[j]:04d}"
        draw.multiline_text((df_fiducials.max_x[j], df_fiducials.centroid_y[j]), label, 255,font)
    imp = add_axes_and_scale_bar(imp, scale_bar_length=50, bin_resolution=config['bin_resolution'])
    # imp.show()
    imfile = os.path.join(config['output_dir'], fiducials_plot_file)
    figure_path = construct_plot_path(imfile, "png", config)
    imp.save(figure_path, quality=95)
    # Extract array with pixels from imp
    im_crop = np.array(imp)
    # remove rows and columns that do not contain text or line segments drawn above
    idx_row = np.sum(im_crop>=line_intensity,axis=1)>0
    idx_col = np.sum(im_crop>=line_intensity,axis=0)>0
    im_crop = im_crop[idx_row][:,idx_col]
    # Write to png
    imfile = os.path.join(config['output_dir'], 'fiducials_plot_cropped')
    figure_path = construct_plot_path(imfile, "png", config)
    imp_crop = Image.fromarray(im_crop)
    imp_crop.save(figure_path, quality=95)
    return 0

def plot_fiducials(df_fiducials: pd.DataFrame, df: pd.DataFrame, config: dict) -> int:
    """
    Plots fiducial summary images and individual fiducial statistics, optionally using multiprocessing.

    Parameters
    ----------
    df_fiducials : pd.DataFrame
        DataFrame containing fiducial information.
    df : pd.DataFrame
        DataFrame containing detection data.
    config : dict
        Configuration dictionary for output settings and multiprocessing options.

    Returns
    -------
    int
        0 on success.
    """
    logging.info("plot_fiducials")
    # Plot summary image first
    plot_fiducial_rois(df_fiducials, df, config)

    nfiducials = len(df_fiducials)
    fiducial_names = df_fiducials['name']
    fiducial_labels = df_fiducials['label']

    if config['multiprocessing']:
        tasks = [(fiducial_labels[j], fiducial_names[j],
                  df[df['label']==fiducial_labels[j]], config) for j in range(nfiducials)]
        with multiprocessing.Pool(int(config['num_threads'])) as pool:
            results = pool.starmap(plot_fiducial, tasks)
    else:
        for j in range(nfiducials):
            fiducial_label = fiducial_labels[j]
            fiducial_name = fiducial_names[j]
            df_detections_roi = df[df['label'] == fiducial_label]
            plot_fiducial(fiducial_label, fiducial_name, df_detections_roi, config)
    return 0

def plot_fiducial(fiducial_label: int, fiducial_name: str, df_detections_roi: pd.DataFrame, config: dict) -> int:
    """
    Plots statistics and projections for a single fiducial region of interest (ROI).

    Parameters
    ----------
    fiducial_label : int
        Label identifying the fiducial ROI.
    fiducial_name : str
        Name of the fiducial ROI.
    df_detections_roi : pd.DataFrame
        DataFrame containing detection data for the ROI.
    config : dict
        Configuration dictionary for output settings and column names.

    Returns
    -------
    int
        0 on success.
    """
    # foreach roi, plot the dependence of z on config['image_id_col'], config['z_step_col'], config['cycle_col'],...
    # Also plot histogram of x,y,x
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    xyz_sd_colnames = [config['x_sd_col'], config['y_sd_col'], config['z_sd_col']]
    dimnames = config['dimnames']
    outdir = os.path.join(config['fiducial_dir'], f"{fiducial_name}")
    os.makedirs(outdir, exist_ok=True)
    logging.info(f"Plotting fiducial {fiducial_name} with label {fiducial_label}")

    x = df_detections_roi[config['x_col']]
    y = df_detections_roi[config['y_col']]
    z = df_detections_roi[config['z_col']]
    deltaz = df_detections_roi[config['deltaz_col']]
    z_step = df_detections_roi[config['z_step_col']]
    n_detections = len(x)

    for k in range(len(dimnames)):
        outpath = os.path.join(outdir, f"{fiducial_name}_{dimnames[k]}_vs_frame")
        image_id = df_detections_roi[config['image_id_col']]
        col_id = xyz_colnames[k]
        vals = df_detections_roi[col_id]
        plot_scatter(image_id, vals, 'image-ID', f'{dimnames[k]} (nm)', f"{dimnames[k]} vs frame", outpath, config)
        plotly_scatter(image_id, vals, None, 'image-ID', f'{dimnames[k]} (nm)', f"{dimnames[k]} vs frame", outpath, config)

    outpath = os.path.join(outdir, f"{fiducial_name}_z_vs_delta_z")
    plot_scatter(deltaz, z, 'delta z (nm)','z (nm)', 'z vs delta z', outpath, config)

    outpath = os.path.join(outdir, f"{fiducial_name}_z_vs_zstep")
    plot_scatter(z_step, z, 'z_step', 'z (nm)', 'z vs zstep', outpath, config)

    # Plot distributions in x,y,z,deltaz
    outpath = os.path.join(outdir, f"{fiducial_name}_hist")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Plot histograms
    axs[0, 0].hist(x, bins=100, color='blue', alpha=0.7)
    axs[0, 0].annotate(stats_text(x, "Summary"), xy=(0.9, 0.9), xycoords='axes fraction', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.5), ha='right', va='top')
    axs[0, 0].set_title('Histogram of x')
    axs[0, 0].set_xlabel('x nm')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(y, bins=100, color='green', alpha=0.7)
    axs[0, 1].annotate(stats_text(y, "Summary"), xy=(0.9, 0.9), xycoords='axes fraction', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.5), ha='right', va='top')
    axs[0, 1].set_title('Histogram of y')
    axs[0, 1].set_xlabel('y nm')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].hist(z, bins=100, color='red', alpha=0.7)
    axs[1, 0].annotate(stats_text(z, "Summary"), xy=(0.9, 0.9), xycoords='axes fraction', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.5), ha='right', va='top')
    axs[1, 0].set_title('Histogram of z')
    axs[1, 0].set_xlabel('z nm')
    axs[1, 0].set_ylabel('Frequency')

    axs[1, 1].hist(deltaz, bins=100, color='purple', alpha=0.7)
    axs[1, 1].set_title('Histogram of delta z')
    axs[1, 1].set_xlabel('delta z nm')
    axs[1, 1].set_ylabel('Frequency')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    # Set point size to suit number of detections
    point_size = 100 / np.max([n_detections,1])
    point_size = np.max([point_size, 0.05])
    point_size = np.min([point_size, 1.0])

    # Plot x,y,z dependence on deltaz
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    outpath = os.path.join(outdir, f"{fiducial_name}_deltaz_dependence")
    figure_path = construct_plot_path(outpath, "png", config)
    colours = ['blue', 'green', 'red']
    for k in range(len(dimnames)):
        col = xyz_colnames[k]
        vals = df_detections_roi[col]
        ax[k].scatter(deltaz, vals, s=point_size, c=colours[k], alpha=0.25)
        # Find the regression line and plot that and the factor and points
        slope, intercept, cor, p_value, std_err = scipy.stats.linregress(deltaz, vals)
        x_fit = np.linspace(np.min(deltaz), np.max(deltaz), 100)
        y_fit = slope * x_fit + intercept
        ax[k].plot(x_fit, y_fit, linestyle=':', color='lightgray')
        # write slope, cor on plot
        ax[k].annotate(f"slope = {slope:.3f}\ncor = {cor:.3f}\n", xy=(0.9, 0.9), xycoords='axes fraction', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.25), ha='right', va='top')
        ax[k].set_xlabel('delta z (nm)')
        ax[k].set_ylabel(f'{col} (nm)')
    plt.savefig(figure_path, dpi=600)
    plt.close()
    # Plot detections colour coded by possible covariates
    columns = [config['image_id_col'], config['z_step_col'], config['cycle_col'], config['time_point_col'],
               config['deltaz_col'], config['photons_col'] ]
    for col in columns:
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        vals = df_detections_roi[col]
        sc = ax[0, 0].scatter(x, z, s=point_size, c=vals, alpha=0.25)
        ax[0, 0].set_xlabel('x (nm)')
        ax[0, 0].set_ylabel('z (nm)')
        ax[0, 1].scatter(y, z, s=point_size, c=vals, alpha=0.25)
        ax[0, 1].set_xlabel('y (nm)')
        ax[0, 1].set_ylabel('z (nm)')
        ax[1, 0].scatter(x, y, s=point_size, c=vals, alpha=0.25)
        ax[1, 0].set_xlabel('x (nm)')
        ax[1, 0].set_ylabel('y (nm)')
        # Use make_axes_locatable to create an inset axis for the colorbar
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes("right", size="15%", pad=0.2)
        cbar = plt.colorbar(sc, cax=cax, label=col)
        cbar.set_alpha(1.0)  # Set colorbar alpha to 1.0 (fully opaque) - otherwise it's too transparent
        cbar.update_normal(sc)  # Update colorbar to apply the alpha setting
        ax[1, 1].set_axis_off()
        outpath = os.path.join(outdir, f"{fiducial_name}_cov_{col}")
        figure_path = construct_plot_path(outpath, "png", config)
        plt.savefig(figure_path, dpi=600)
        plt.close()
    # Plot x,y,z vs x_sd, y_sd, z_sd  to see if error estimate is realistic
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    for k in range(len(dimnames)):
        x_col_id = xyz_colnames[k]
        x = df_detections_roi[x_col_id]
        y_col_id = xyz_sd_colnames[k]
        y = df_detections_roi[y_col_id]
        ax[k].scatter(x, y, s=point_size, c=colours[k], alpha=0.25)
        # Possibly interesting but not generally applicable, so commented out
        do_fit=False
        if do_fit:
            # Find the regression line and plot that and the factor and points on the points
            slope, intercept, cor, p_value, std_err = scipy.stats.linregress(x, y)
            x_fit = np.linspace(np.min(x), np.max(x), 100)
            y_fit = slope * x_fit + intercept
            ax[k].plot(x_fit, y_fit, linestyle=':', color='lightgray')
            # write slope, cor, std_err on plot
            ax[k].annotate(f"slope = {slope:.3f}\ncor = {cor:.3f}\n", xy=(0.9, 0.9), xycoords='axes fraction', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.25), ha='right', va='top')
        ax[k].set_xlabel(f'{x_col_id} (nm)')
        ax[k].set_ylabel(f'{y_col_id} (nm)')
    outpath = os.path.join(outdir, f"{fiducial_name}_sd")
    figure_path = construct_plot_path(outpath, "png", config)
    plt.savefig(figure_path, dpi=600)
    plt.close()

    return 0


def plot_fiducial_quality_metrics(df_fiducials: pd.DataFrame, config: dict):
    """
    Plots quality metrics for fiducials, including statistics and boxplots.

    Parameters
    ----------
    df_fiducials : pd.DataFrame
        DataFrame containing fiducial statistics.
    config : dict
        Configuration dictionary for output settings and column names.

    Returns
    -------
    None
        Saves the plots to file.
    """
    dimnames =config['dimnames']
    ndim = len(dimnames)
    quantities = ['deltaz_slope','deltaz_cor', 'sd', 'fwhm', 'z_step_cor']
    units = ['', 'nm', 'nm', '']

    for unit,fiducial_stat in zip (units, quantities):
        logging.info(f"Plotting fiducial stat {fiducial_stat}")
        outdir = os.path.join(config['fiducial_dir'])
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f"{fiducial_stat}_vs_xyz")
        fig, axs = plt.subplots(3, 3, figsize=(12, 10))
        for j in range(len(dimnames)):
            for k in range(len(dimnames)):
                y_col = f"{dimnames[k]}_{fiducial_stat}"
                x_col = f"{dimnames[j]}_mean"
                x = df_fiducials[x_col]
                y = df_fiducials[y_col]
                axs[j, k].set_title(f"{y_col} vs {x_col}")
                axs[j, k].set_xlabel(f"{x_col} (nm)")
                axs[j, k].set_ylabel(f"{y_col} {unit}")
                axs[j, k].scatter(x,y)
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()

    # plot photons versus x_mean, y_mean, z_mean in a 3 panel plot
    outdir = os.path.join(config['fiducial_dir'])
    outpath = os.path.join(outdir, "photons_vs_xyz")
    fig, axs = plt.subplots(3, 1, figsize=(12, 4))
    for j in range(ndim):
        x_col = f"{dimnames[j]}_mean"
        x = df_fiducials[x_col]
        y = df_fiducials['photons_mean']
        axs[j].scatter(x, y)
        axs[j].set_title(f"mean photons vs {x_col}")
        axs[j].set_xlabel(f"{x_col} (nm)")
        axs[j].set_ylabel("photons")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
    outpath = os.path.join(config['output_dir'], f"quality_metrics_summary.png")
    plt.figure(figsize=(10, 6))
    outdir = config['output_dir']
    os.makedirs(outdir, exist_ok=True)
    sd_metrics = np.column_stack((
        df_fiducials['x_sd'].values,
        df_fiducials['y_sd'].values,
        df_fiducials['z_sd'].values
    ))
    # box plots of SD's for all fiducials
    plt.boxplot(sd_metrics, labels=xyz_colnames)
    plt.xlabel('quantity')
    plt.ylabel(f"SD (nm)")
    plt.title(f'SD of fiducials')
    plt.savefig(outpath)
    plt.close()

def plot_drift_correction(df_drift: pd.DataFrame, config: dict):
    """
    Plots drift correction statistics for x, y, z dimensions over time.

    Parameters
    ----------
    df_drift : pd.DataFrame
        DataFrame containing drift correction data.
    config : dict
        Configuration dictionary for output settings and column names.

    Returns
    -------
    None
        Saves the plots to file.
    """
    x_col = ['x', 'y', 'z']
    xsd_col = ['x_sd', 'y_sd', 'z_sd']
    ndims = len(x_col)

    # Plot the drift correction with error bars
    for j in range(ndims):
        # output_path = os.path.join(config['output_dir'], f"drift_correction_{x_col[j]}")
        # plotly_scatter(df_drift['image-ID'], df_drift[x_col[j]], df_drift[xsd_col[j]], 'image-ID', f'{x_col[j]} correction (nm)', 'Drift correction', output_path, config)
        outpath = os.path.join(config['output_dir'], f"corrections_{x_col[j]}_vs_time")
        plot_scatter(df_drift[config['image_id_col']], df_drift[x_col[j]], config['image_id_col'], f'{x_col[j]} correction (nm)',
                     f'Correction {x_col[j]} vs image-ID', outpath, config)

def plot_time_derivatives(df_drift: pd.DataFrame, config: dict):
    """
    Plots time derivatives of drift correction for x, y, z dimensions.

    Parameters
    ----------
    df_drift : pd.DataFrame
        DataFrame containing drift correction data.
    config : dict
        Configuration dictionary for output settings and column names.

    Returns
    -------
    None
        Saves the plots to file.
    """
    x_col = ['x', 'y', 'z']
    xsd_col = ['x_sd', 'y_sd', 'z_sd']
    ndims = len(x_col)

    # Plot the drift correction with error bars
    for j in range(ndims):
        outpath = os.path.join(config['output_dir'], f"d{x_col[j]}dt_vs_time")
        plot_scatter(df_drift[config['image_id_col']], df_drift[f'd{x_col[j]}_dt'], config['image_id_col'], f'd{x_col[j]}_dt (nm/timepoint)',
                     f'Correction {x_col[j]} vs image-ID', outpath, config)

def save_to_tiff_3d(counts_xyz: np.ndarray, filename: str, config: dict):
    """
    Saves a 3D numpy array to a TIFF file, choosing data type based on pixel values.

    Parameters
    ----------
    counts_xyz : np.ndarray
        3D array of pixel values to save.
    filename : str
        Filename for the TIFF file.
    config : dict
        Configuration dictionary for output settings.

    Returns
    -------
    None
        Saves the TIFF file to disk.
    """
    logging.info("save_to_tiff_3d")
    # Save a 3D array to a TIFF file
    maxpixel = np.max(counts_xyz)
    imgfile = construct_plot_path(filename, "tif", config)
    logging.info(f"Saving as binned image to {imgfile}. Max pixel value: {maxpixel}.")
    # Reverse the order of the dimensions indices to match the order of the axes in the image
    img =  np.transpose(counts_xyz, (2, 1, 0))

    if maxpixel < 256:
        tifffile.imwrite(imgfile, img.astype(np.uint8), imagej=True)
    elif maxpixel < 16384:
        tifffile.imwrite(imgfile, img.astype(np.uint16), imagej=True)
    else:
        tifffile.imwrite(imgfile, img.astype(np.float32), imagej=True)

def save_to_tiff_2d(img: np.ndarray, filename: str, config: dict):
    """
    Saves a 2D numpy array to a TIFF file, choosing data type based on pixel values.

    Parameters
    ----------
    img : np.ndarray
        2D array of pixel values to save.
    filename : str
        Filename for the TIFF file.
    config : dict
        Configuration dictionary for output settings.

    Returns
    -------
    None
        Saves the TIFF file to disk.
    """
    logging.info("save_to_tiff_2d")
    # Save a 3D array to a TIFF file
    maxpixel = np.max(img)
    imgfile = construct_plot_path(filename, "tif", config)
    logging.info(f"Saving as binned image to {imgfile}. Max pixel value: {maxpixel}.")
    if maxpixel < 256:
        tifffile.imwrite(imgfile, img.astype(np.uint8), imagej=True)
    elif maxpixel < 16384:
        tifffile.imwrite(imgfile, img.astype(np.uint16), imagej=True)
    else:
        tifffile.imwrite(imgfile, img.astype(np.float32), imagej=True)
