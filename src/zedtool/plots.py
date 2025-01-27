#!/usr/bin/env python3
import logging
import plotly.express as px

import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile
from PIL import Image, ImageDraw, ImageFont
from zedtool.srxstats import extract_z_correction, z_means_by_marker

# Prints some debugging plots for an SRX dataset.
# Takes a corrected and an uncorrected table of detections, registers the rows and finds the corrections.
# Write out a table with both corrected and uncorrected z.

def construct_plot_path(filename: str, filetype: str, config: dict) -> str:
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
            figure_dir = os.path.join(dir_path, "plots")
            figure_path = os.path.join(dir_path, "plots", file_name)
    # If figure_dir doesn't exist, create it
    figure_dir = os.path.dirname(figure_path)
    os.makedirs(figure_dir, exist_ok=True)
    return figure_path

def plot_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, filename: str, config: dict) -> int:
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
    filetype = "html"
    figure = px.scatter(x=x, y=y, error_y=y_err, title=title, labels={xlabel: xlabel, ylabel: ylabel})
    # if filename doesn't contain config['output_dir'], prepend it
    figure_path = construct_plot_path(filename, filetype, config)
    figure.write_html(figure_path)
    return 0

def plot_histogram(x: np.ndarray, xlabel: str, ylabel: str, title: str, filename: str, config: dict) -> int:
    hist_bins = 100 # TODO: make this a config option
    filetype = "png"
    plt.figure()
    plt.hist(x, bins=hist_bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    figure_path = construct_plot_path(filename, filetype, config)
    plt.savefig(figure_path, dpi=300)
    plt.close()
    return 0

def plot_detections(df: np.ndarray, filename: str, config: dict) -> int:
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
    figure_path = construct_plot_path("detections_summary", "png", config)
    plt.savefig(figure_path, dpi=600)
    plt.close()

def plot_binned_detections_stats(n_xy: np.ndarray,mean_xy: np.ndarray, sd_xy: np.ndarray, filename: str, config: dict) -> int:
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
    figure_path = construct_plot_path("binned_detections_summary", "png", config)
    fig.savefig(figure_path, dpi=600)
    plt.close()

def plot_summary_stats(df: np.ndarray, det_xyz: np.ndarray, config: dict) -> int:
    # Plot detections and other quantities
    z_step_step = config['z_step_step']

    # plot_histogram(df[config['z_step_col']], 'z-step', 'Detections', "Detections by z-step", "zstep_histogram", df['z-step'].max()-df['z-step'].min()+1, config)
    plot_histogram(df[config['z_step_col']], 'z-step', 'Detections', "Detections by z-step", "zstep_histogram", config)
    plot_histogram(df[config['z_col']] - df[config['z_step_col']] * z_step_step, 'z - z-step*z_step_step (nm)', 'Detections',
                   'Detections by z-z-step*z_step_step', 'z_zstep_histogram', config)
    plot_scatter(df[config['image_id_col']], df[config['z_col']], 'image-ID', 'z (nm)', 'z vs frame', 'z_vs_frame', config)
    plot_scatter(df[config['image_id_col']], df[config['photons_col']], 'image-ID', 'photon-count', 'photon-count vs frame',
                 'photon_count_vs_frame', config)
    plot_scatter(df[config['image_id_col']], df[config['z_step_col']], 'image-ID', 'z-step', 'z-step vs frame',
                 'zstep_vs_frame', config)
    plot_scatter(df[config['z_col']] - df[config['z_step_col']] * z_step_step, df[config['z_col']],
                 'z - z-step*z_step_step (nm)', 'z (nm)', 'z vs z - z-step*z_step_step', 'z_vs_z_zstep', config)

    z_mean, t = z_means_by_marker(det_xyz, df[config['image_id_col']].values)
    plot_scatter(t, z_mean, 'time', 'mean(z) per frame (nm)', 'mean(z) vs time', 'z_mean_per_frame_vs_time', config)

    z_mean, cycle = z_means_by_marker(det_xyz, df[config['cycle_col']].values)
    plot_scatter(cycle, z_mean, 'cycle', 'mean(z) per cycle (nm)', 'mean(z) vs cycle', 'z_mean_per_cycle_vs_cycle', config)

    z_mean, z_step = z_means_by_marker(det_xyz, df[config['z_step_col']].values)
    plot_scatter(z_step, z_mean, 'z-step', 'mean(z) per z-step (nm)', 'mean(z) vs z-step', 'z_mean_per_z_step_vs_z_step',
                 config)

def plot_fiducials(df_fiducials: np.ndarray, df: np.ndarray, config: dict) -> int:
    #   * plot z vs time, projections coloured by quantities, dendrogram of groupings
    logging.info("plot_fiducials")
    detections_img_file = 'detections_img.tif'
    fiducials_plot_file = 'fiducials_plot'
    # read in the image and the segmentation from tifs
    img_filt = tifffile.imread(os.path.join(config['output_dir'], detections_img_file))
    # scale the image to 0-255
    im = 255 * (np.log10(img_filt+1)/np.log10(np.max(img_filt)+1))
    im = im.astype(np.uint8)
    # loop over rows of df
    for j in range(len(df_fiducials)):
        im[int(df_fiducials.min_y[j]):int(df_fiducials.max_y[j]-1), int(df_fiducials.min_x[j])] = 228
        im[int(df_fiducials.min_y[j]):int(df_fiducials.max_y[j]-1), int(df_fiducials.max_x[j]-1)] = 228
        im[int(df_fiducials.min_y[j]), int(df_fiducials.min_x[j]):int(df_fiducials.max_x[j]-1)] = 228
        im[int(df_fiducials.max_y[j]-1), int(df_fiducials.min_x[j]):int(df_fiducials.max_x[j]-1)] = 228
    imp = Image.fromarray(im)
    draw = ImageDraw.Draw(imp)
    for fontname in ["DejaVuSansMono", "couri", "Courier New", "arial", "LiberationSans-Regular"]:
        try:
            font = ImageFont.truetype(fontname, 10)
            break
        except OSError:
            pass
    # Add text
    for j in range(len(df_fiducials)):
        label = df_fiducials.label[j]
        draw.text((df_fiducials.max_x[j], df_fiducials.centroid_y[j]), f"f_{label:04d}", 255,font)
    # imp.show()
    imfile = os.path.join(config['output_dir'], fiducials_plot_file)
    figure_path = construct_plot_path(imfile, "png", config)
    imp.save(figure_path, quality=95)

    # foreach roi, plot the dependence of z on config['image_id_col'], config['z_step_col'], config['cycle_col']
    # Also plot histogram of x,y,x
    for j in range(len(df_fiducials)):
        # columns = [config['image_id_col'], config['z_step_col'], config['frame_col'], config['time_point_col'], config['cycle_col']]
        columns = [config['image_id_col'], config['z_step_col'], config['cycle_col'], config['time_point_col']]
        fiducial_label = df_fiducials.at[j, 'label']
        fiducial_name = df_fiducials.at[j, 'name']
        outdir = os.path.join(config['fiducial_dir'], f"{fiducial_name}")
        os.makedirs(outdir, exist_ok=True)
        logging.info(f"Plotting fiducial {fiducial_name} with label {fiducial_label}")
        df_detections_roi = df[df['label'] == fiducial_label]
        x = df_detections_roi[config['x_col']]
        y = df_detections_roi[config['y_col']]
        z = df_detections_roi[config['z_col']]
        z_step = df_detections_roi[config['z_step_col']]
        z_step_step = config['z_step_step']
        adjusted_z = z - z_step * z_step_step

        outpath = os.path.join(outdir, f"{fiducial_name}_z_vs_frame")
        image_id = df_detections_roi[config['image_id_col']]
        plot_scatter(image_id, z, 'image-ID', 'z (nm)', "z_vs_frame", outpath, config)
        plotly_scatter(image_id, z, None, 'image-ID', 'z (nm)', "z_vs_frame", outpath, config)

        outpath = os.path.join(outdir, f"{fiducial_name}_z_vs_z_zstep")
        plot_scatter(adjusted_z, z, 'z - z-step*z_step_step (nm)','z (nm)', 'z vs z - z-step*z_step_step', outpath, config)

        outpath = os.path.join(outdir, f"{fiducial_name}_z_vs_zstep")
        plot_scatter(z_step, z, 'z_step', 'z (nm)', 'z vs zstep', outpath, config)

        outpath = os.path.join(outdir, f"{fiducial_name}_hist")
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        # Plot histograms
        axs[0, 0].hist(x, bins=100, color='blue', alpha=0.7)
        axs[0, 0].set_title('Histogram of x')
        axs[0, 0].set_xlabel('x nm')
        axs[0, 0].set_ylabel('Frequency')

        axs[0, 1].hist(y, bins=100, color='green', alpha=0.7)
        axs[0, 1].set_title('Histogram of y')
        axs[0, 1].set_xlabel('y nm')
        axs[0, 1].set_ylabel('Frequency')

        axs[1, 0].hist(z, bins=100, color='red', alpha=0.7)
        axs[1, 0].set_title('Histogram of z')
        axs[1, 0].set_xlabel('z nm')
        axs[1, 0].set_ylabel('Frequency')

        axs[1, 1].hist(adjusted_z, bins=100, color='purple', alpha=0.7)
        axs[1, 1].set_title('Histogram of z - z_step * z_step_step')
        axs[1, 1].set_xlabel('Adjusted z - z_step * z_step_step  nm')
        axs[1, 1].set_ylabel('Frequency')
        # Adjust layout
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()

        for col in columns:
            fig, ax = plt.subplots(2, 2, figsize=(12, 9))
            sc = ax[0, 0].scatter(x, z, s=0.05, c=df_detections_roi[col], alpha=0.25)
            ax[0, 0].set_xlabel('x (nm)')
            ax[0, 0].set_ylabel('z (nm)')
            ax[0, 1].scatter(y, z, s=0.05, c=df_detections_roi[col], alpha=0.25)
            ax[0, 1].set_xlabel('y (nm)')
            ax[0, 1].set_ylabel('z (nm)')
            ax[1, 0].scatter(x, y, s=0.05, c=df_detections_roi[col], alpha=0.25)
            ax[1, 0].set_xlabel('x (nm)')
            ax[1, 0].set_ylabel('y (nm)')
            # Use make_axes_locatable to create an inset axis for the colorbar
            divider = make_axes_locatable(ax[1, 1])
            cax = divider.append_axes("right", size="15%", pad=0.2)
            cbar = plt.colorbar(sc, cax=cax, label=col)
            ax[1, 1].set_axis_off()
            outpath = os.path.join(outdir, f"{fiducial_name}_{col}")
            figure_path = construct_plot_path(outpath, "png", config)
            plt.savefig(figure_path, dpi=600)
            plt.close()
    return 0

