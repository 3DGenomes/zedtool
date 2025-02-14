#!/usr/bin/env python3
import logging
import plotly.express as px
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tifffile
from PIL import Image, ImageDraw, ImageFont
from zedtool.srxstats import z_means_by_marker
from zedtool.detections import fwhm_from_points

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
    figure_path = construct_plot_path(filename, "png", config)
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
    figure_path = construct_plot_path(filename, "png", config)
    fig.savefig(figure_path, dpi=600)
    plt.close()

def plot_summary_stats(df: np.ndarray, det_xyz: np.ndarray, config: dict) -> int:
    # Plot detections and other quantities

    # plot_histogram(df[config['z_step_col']], 'z-step', 'Detections', "Detections by z-step", "zstep_histogram", df['z-step'].max()-df['z-step'].min()+1, config)
    plot_histogram(df[config['z_step_col']], 'z-step', 'Detections', "Detections by z-step", "zstep_histogram", config)
    plot_histogram(df[config['deltaz_col']], f"{config['deltaz_col']} (nm)", 'Detections',
                   'Detections delta z', 'delta_z_histogram', config)
    plot_scatter(df[config['image_id_col']], df[config['z_col']], 'image-ID', 'z (nm)', 'z vs frame', 'z_vs_frame', config)
    plot_scatter(df[config['image_id_col']], df[config['photons_col']], 'image-ID', 'photon-count', 'photon-count vs frame',
                 'photon_count_vs_frame', config)
    plot_scatter(df[config['image_id_col']], df[config['z_step_col']], 'image-ID', 'z-step', 'z-step vs frame',
                 'zstep_vs_frame', config)
    plot_scatter(df[config['deltaz_col']] , df[config['z_col']],
                 'delta z (nm)', 'z (nm)', 'z vs delta z', 'z_vs_delta_z', config)

    z_mean, t = z_means_by_marker(det_xyz, df[config['image_id_col']].values)
    plot_scatter(t, z_mean, 'time', 'mean(z) per frame (nm)', 'mean(z) vs time', 'z_mean_per_frame_vs_time', config)

    z_mean, cycle = z_means_by_marker(det_xyz, df[config['cycle_col']].values)
    plot_scatter(cycle, z_mean, 'cycle', 'mean(z) per cycle (nm)', 'mean(z) vs cycle', 'z_mean_per_cycle_vs_cycle', config)

    z_mean, z_step = z_means_by_marker(det_xyz, df[config['z_step_col']].values)
    plot_scatter(z_step, z_mean, 'z-step', 'mean(z) per z-step (nm)', 'mean(z) vs z-step', 'z_mean_per_z_step_vs_z_step',
                 config)

def stats_text(x: np.ndarray,title: str) -> str:
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

def plot_fiducials(df_fiducials: np.ndarray, df: np.ndarray, config: dict) -> int:
    #   * plot z vs time, projections coloured by quantities, dendrogram of groupings
    logging.info("plot_fiducials")
    detections_img_file = 'detections_img.tif'
    fiducials_plot_file = 'fiducials_plot'
    dimnames = config['dimnames']
    xyz_colnames = [config['x_col'], config['y_col'], config['z_col']]
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
        columns = [config['image_id_col'], config['z_step_col'], config['cycle_col'], config['time_point_col'], config['deltaz_col']]
        fiducial_label = df_fiducials.at[j, 'label']
        fiducial_name = df_fiducials.at[j, 'name']
        outdir = os.path.join(config['fiducial_dir'], f"{fiducial_name}")
        os.makedirs(outdir, exist_ok=True)
        logging.info(f"Plotting fiducial {fiducial_name} with label {fiducial_label}")
        df_detections_roi = df[df['label'] == fiducial_label]
        x = df_detections_roi[config['x_col']]
        y = df_detections_roi[config['y_col']]
        z = df_detections_roi[config['z_col']]
        deltaz = df_detections_roi[config['deltaz_col']]
        z_step = df_detections_roi[config['z_step_col']]

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

        # Plot x,y,z dependence on deltaz
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        outpath = os.path.join(outdir, f"{fiducial_name}_deltaz_dependence")
        figure_path = construct_plot_path(outpath, "png", config)
        ax[0].scatter(deltaz, x, s=0.05, c='blue', alpha=0.25)
        ax[0].set_xlabel('delta z (nm)')
        ax[0].set_ylabel('x (nm)')
        ax[1].scatter(deltaz, y, s=0.05, c='green', alpha=0.25)
        ax[1].set_xlabel('delta z (nm)')
        ax[1].set_ylabel('y (nm)')
        ax[2].scatter(deltaz, z, s=0.05, c='red', alpha=0.25)
        ax[2].set_xlabel('delta z (nm)')
        ax[2].set_ylabel('z (nm)')
        plt.savefig(figure_path, dpi=600)
        plt.close()
        # Plot detections colour coded by possible covariates
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

def plot_fiducial_quality_metrics(df_fiducials: np.ndarray, config: dict) -> int:
    dimnames =config['dimnames']
    ndim = len(dimnames)
    quantities = ['deltaz_cor', 'sd', 'fwhm']
    units = ['', 'nm', 'nm']
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

def save_to_tiff_3d(counts_xyz: np.ndarray, filename: str, config: dict):
    # Save a 3D array to a TIFF file
    maxpixel = np.max(counts_xyz)
    imgfile = construct_plot_path(filename, "tif", config)
    logging.info(f"Saving as binned image to {imgfile}. Max pixel value: {maxpixel}.")
    # Reverse the order of the dimensions indices to match the order of the axes in the image
    img =  np.transpose(counts_xyz, (2, 1, 0))

    if maxpixel < 256:
        tifffile.imsave(imgfile, img.astype(np.uint8), imagej=True)
    elif maxpixel < 16384:
        tifffile.imsave(imgfile, img.astype(np.uint16), imagej=True)
    else:
        tifffile.imsave(imgfile, img.astype(np.float32), imagej=True)


