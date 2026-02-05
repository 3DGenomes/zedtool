# Overview
*Z* *E*stimate *D*iagnostic tool (zedtool) - a tool for evaluating and processing detections from SMLM experiments.

## Prerequisites
zedtool requires Python 3.12 or higher and miniconda/conda-forge for managing dependencies.

To install python, follow the instructions at https://www.python.org/downloads/

To install miniconda, follow the instructions at https://www.anaconda.com/docs/getting-started/miniconda/install

## Installation

To install zedtool from source:

```bash
git clone https://github.com/3DGenomes/zedtool.git
cd zedtool
conda env create --file=environment.yaml
conda activate zedtool-env
pip install .
```
An alternative to git clone is to download the source code as a zip file from the GitHub repository.
In this case, unzip the file and navigate to the resulting directory and proceed as before.
For example:

```bash
unzip zedtool-master.zip
cd zedtool-master
...
```

## Running zedool

Once installed, from within the conda environment, you can run zedtool using:

```bash
zedtool config.yaml
```

where `config.yaml` is a YAML configuration file containing experiment-specific parameters.

## Configuration Options
All options for zedool are supplied in a YAML configuration file specified on the command line. 
Below is a detailed explanation of the parameters used in the file with default values in parentheses.
For an overview of the workflow and example output see [here](docs/source/manual.md).

### **Input and Output**
- `detections_file`: Path to the input file containing detections. Can be any object that can be opened by 
the fsspec library, such as a local file, a file on S3, or a file on Google Cloud Storage.
If it ends in `gz` then it will be decompressed. (`input.csv`)
- `output_dir`: Directory where all output files will be stored. (`output`)

### **Binning and Debugging Options**
- `bin_resolution`: Bin size in nm for the binned image. (`20`)
- `z_step_step`: Step size in nm for one step in the Z direction. Sign must match microscope setting. (`-100`)
- `debug`: If set to 1, prints extra output and maks extra files. (`0`)
- `verbose`: If set to 1, prints extra output to the terminal. (`1`)
- `multiprocessing`: If 1 then use multiprocessing to speed up the computation. (`0`)
- `num_threads`: Number of threads to use for multiprocessing. If zero then uses all available. (`0`)
- `float_format`: printf-style format for floating point numbers in outputs files. (`%.6g`)
- `ignore_image_id_col`: If 1, ignore the image_id_col column in the input file. If dropped it will be regenerated. (`0`)
- `create_backup_columns`: If 1 then backup x_col,y_col,z_col, x_sd_col,y_sd_col,z_sd_col before changing them. (`0`)
- `use_pyarrow`: If 1, use pyarrow for reading/writing tabular data. This is faster but requires more memory. (`0`)

### **Detection Filtering**
- `select_cols`: Comma-separated list of selection criteria columns. All criteria are *and*-ed together. 
This can be used for quality filtering and or for processing only a portion of the field of view. (``)
- `select_ranges`: Comma-separated list of value ranges for filtering detections. 
Number of entries must match select_cols. (`0-0`)
- `threshold_on_density`: If 1 then threshold detections based on density in binned image. (`0`)
- `threshold_dimensions`: Dimensions used for binning the image used for thresholding (2 or 3). (`2`)
- `threshold_min_cutoff`: Minimum detections in a bin for thresholding. (`1`)
- `threshold_max_cutoff`: Maximum detections in a bin for thresholding. (`1e10`)

### **Frame, z-step, cycle, and time-point Ranges**
- `frame_range`: Range of frames per z-step. (`0-0`)
- `z_step_range`: Range of z-steps per cycle. (`0-0`)
- `cycle_range`: Number of cycles per time-point. (`0-0`)
- `time_point_range`: Number of time-points per experiment. (`0-0`)

### **Deconvolution parameters**
- `decon_min_cluster_sd`: Minimum standard deviation of clusters to be considered for deconvolution. (`10`)
- `decon_sd_shrink_ratio`: Ratio of final/initial standard deviation of clusters following deconvolution. (`0.25`)
- `decon_bin_threshold`: Minimum x-y bin threshold to be considered for deconvolution. (`100`)
- `decon_kmeans_max_k`: Maximum number of clusters in z to be considered for deconvolution. (`5`)
- `decon_kmeans_proximity_threshold`: Clusters closer than (SD1 + SD2) * proximity_threshold are merged. (`2.0`)
- `decon_kmeans_min_cluster_detections`: Minimum number of detections in a cluster to be considered for deconvolution. (`5`)

### **Column Names in Input Data**
These specify how different columns in the detections file are named:

- `frame_col` (`frame`), `image_id_col` (`image-ID`), `z_step_col` (`z-step`), `cycle_col` (`cycle`), `time_point_col (`time-point`)`: 
Identifiers for frames, image-ids, z-steps, cycles, and time-points.
- `x_col` (`x`), `y_col` (`y`), `z_col` (`z`): Spatial coordinates.
- `x_sd_col` (`precisionx`), `y_sd_col` (`precisiony`), `z_sd_col` (`precisionz`): Precision values for x, y, and z coordinates.
- `photons_col`: Photon count. (`photon-count`)
- `chisq_col`: Chi-squared value of the fit (not currently used but may be in future versions). (`chisq`)
- `deltaz_col`: Distance of z relative to focal plane or putative biplane midpoint. (`deltaz`)
- `log_likelihood_col`: log likelihood of the fit - currently not used. (`log-likelihood`)
- `llr_col`: llr log likelihood ratio of the fit - currently not used. (`llr`)
- `probe_col`: probe ID when more than one channel is present. (`vis-probe`)

### **Fiducial Segmentation and Filtering**
- `excluded_fiducials`: Comma-separated list of fiducials to exclude from drift correction. (`None`)
- `included_fiducials`: Comma-separated list of fiducials to use for drift correction (use _only_ these). (`None`)
- `resegment_after_correction:` If set to 1, re-segment the fiducials after drift correction. (`0`)
This improves the accuracy of the fiducial measurements but potentially changes their labels, their by invalidating
comparisons of pre- and post-correction fiducial positions.
- `median_filter_disc_radius`: Disc radius for filtering out speckles in the binned image. (`1`)
- `filling_disc_radius`: Radius for filling holes in thresholded images. (`10`)
- `dilation_disc_radius`: Dilation radius for catching "wandering" fiducials. (`10`)
- `min_fiducial_size`: Minimum size (in pixels) of fiducials. (`100`)
- `min_fiducial_detections`: Minimum number of detections required for a fiducial. (`10`)
- `max_detections_per_image`: Maximum number of detections allowed per fiducial per image. (`1.1`)
- `quantile_outlier_cutoff`: Discarded fraction of extreme values for quality control (typically 0.01-0.05). If you have less than
20 fiducials or are selecting them manually then set this to 0. (`0`)
- `sd_outlier_cutoff`: Discard fiducials that are more than this many standard deviations away from the mean.
If you have less than 20 fiducials or are selecting them manually then set this to 0. (`0`)
- `filter_fiducials_with_clustering`: If 1, then cluster on intensity in 2D binned image and select "high" cluster. 
Not very robust. Use only if all else fails. (`0`)
- `filter_columns`: Comma separated list of columns to use for filtering fiducials.
(`mean_intensity, n_detections, x_sd, y_sd, z_sd, photons_mean, x_madr, y_madr, z_madr, time_point_separation`)
- `polynomial_degree`: Polynomial degree for drift correction fitting. Best left at 1 or 2. (`2`)
- `fitting_interval`: Interval over which to fit the polynomial curve to the drift correction - [frame|z_step|cycle|time_point]. (`time_point`)
- `use_weights_in_fit`: Whether to use uncertainty values for x,y,z when fitting. (`0`)
- `minimum_detections_for_fit`: Minimum number of fiducial detections required per fitting interval. (`1000`)
- `only_fiducials`: If set to 1, assumes all "bright spots" in the image are fiducials. 
If you want to find the fiducials automatically then set this to 0. (`0`)
- `consensus_method`: Method for determining the consensus z value for a fiducial. Options are 'weighted_mean' and 'median'. (`median`)
- `refilter_fiducials_after_correction`: If set to 1, refilter fiducials after initial drift correction. 
Not recommended because it invalidates the comparison between metrics before and after correction. (`0`)

### **Appending another experiment**
Done before all other operations when `concatenate_detections` is set.

- `concatenate_detections_file`: file to concatenate on the end of the current experiment. (`None`)
- `concatenate_offset_file`: offsets to apply to the detections in the concatenate_detections_file. (`None`)
The file `concatenate_offset_file` contains one row with the offsets to add.
Aside from `image_id_col`, `time_point_col` and `x_col`, `y_col`, `z_col`, columns, all values should be zero.
The columns `image_id_col` and `time_point_col` are used to make the corresponding columns in the output file consistent.
The columns  `x_col`, `y_col` and `z_col` are the spatial offsets _added_ to the detections in the `concatenate_detections_file`.

### **Applying a pre-computed drift correction**
- `drift_correction_file`: File containing drift correction values. (`drift_correction.tsv`)
Used if `apply_drift_correction` is enabled.
If `apply_drift_correction` is set then the drift corrections in `drift_correction_file` are applied.
This is done directly after any concatenation of other detections but before any other processing.
The values in the column `image_id_col` in `drift_correction_file` must match.

### **Computation and Plotting Steps**
You can disable/enable various processing steps and visualisations by setting these values to 0/1. 
These are listed below in the order in which they are performed.

- `concatenate_detections`: Concatenate all detections from concatenate_detections_file using offsets from concatenate_offset_file. (`0`)
- `apply_drift_correction`: Apply pre-made drift corrections from `drift_correction_file` to detections. 
If `correct_detections` is set then a drift correction file is written out that can be applied to a set of detections from the same experiment.(`0`)
- `plot_detections`: Plot all detections as binned x-y images and 3D projections. (`0`)
- `threshold_on_density`: Remove detections based on density in the binned image. Can be useful to remove background that can drag down the threshold for segmentation. (`0`)
- `make_quality_metrics`: Compute quality metrics for fiducials before correction. Quality metrics are re-computed at the end after all corrections. (`0`)
- `plot_fiducial_correlations`: Plot correlations between fiducials. Normally not necessary. Can be slow! (`0`)
- `plot_fiducials`: Plot fiducials before any correction. (`0`)
- `plot_summary_stats`: Generate summary statistics plots for detections. (`0`)
- `plot_time_point_metrics`: Plot time-point metrics of drift correction for fiducials. (`0`)
- `plot_per_fiducial_fitting`: Make debugging images showing extraction of drift correction from fiducials at each time step (`0`)
- `plot_fourier_correlation`: Plot Fourier Ring/Shell Correlation for the 2D/3D binned image. (`0`)
- `rotation_correct_detections`: Correct for rotation/translation of the sample at time point boundaries.  (`0`)
- `drift_correct_detections`: Do fiducial-based drift correction on all detections. This changes the x,y,z,... columns and copies them to x_0,y_0,... 
In addition, it writes a file to the output directory called `drift_correction.tsv` that contains these corrections that can be used as described above.  (`0`)
- `drift_correct_detections_multi_pass`: Do second drift correction pass. Re-filters fiducials after initial correction and re-computes drift correction. (`0`)
- `deltaz_correct_detections`: Correct z co-ordinate of all detections for deltaz variation. (`0`)
- `deconvolve_z`: Reduce variation in z by squeezing peaks found in x-y bins. Experimental.  (`0`)
- `save_non_fiducial_detections` Save corrected non-fiducial detections to a separate file, `corrected_detections_no_fiducials.csv`  (`0`)
- `save_fiducial_detections` Save corrected fiducial detections to a separate file, `corrected_detections_fiducials.csv`  (`0`)
- `create_backup_columns` If set to 1 then backup x_col,y_col,z_col, x_sd_col,y_sd_col,z_sd_col before changing them. (`0`)

### **Computation and Plotting Options**
- `covariate_plot_quantities`: List of columns to plot as per-fiducial covariates (`image_id_col`, `z_step_col`, `cycle_col`, `time_point_col`, `deltaz_col`, `photons_col`, `x_sd_col`, `y_sd_col`, `z_sd_col`)
- `plot_format`: File type to be used for plotting - [png|jpg|pdf|svg|...]. (`png`)
- `plot_dpi`: DPI to use for raster plots (`150`)
## Example Configuration File
Options are read from a yaml configuration file supplied on the command line. Here is an example:

```yaml
# Files and paths
detections_file: /path/to/input.csv 
output_dir: /path/to/output 

# Experiment details
frame_range: 0-199 
z_step_range: 0-60 
cycle_range: 0-4 
time_point_range: 0-0 

# Binning and debugging
bin_resolution: 20 
z_step_step: -100 

# Detection filtering
select_cols: x,y 
select_ranges: 0-10000,20000-30000 

# Column names specific to the input file format
# The ones below are for Bruker's SRX software
frame_col: frame 
image_id_col: image-ID 
z_step_col: z-step
cycle_col: cycle
time_point_col: time-point
x_col: x
y_col: y
z_col: z
x_sd_col: precisionx
y_sd_col: precisiony
z_sd_col: precisionz
photons_col: photon-count
chisq_col: chisq 
deltaz_col: deltaz

# Constants used for segmenting, filtering and fitting
excluded_fiducials: 
median_filter_disc_radius: 1 
filling_disc_radius: 10 
dilation_disc_radius: 10 
min_fiducial_size: 100
min_fiducial_detections: 10 
max_detections_per_image: 1.1 
sd_outlier_cutoff: 2 
polynomial_degree: 2 
only_fiducials: 0 

# What steps to perform
plot_summary_stats: 1
plot_detections: 1 
plot_fiducials: 1 
drift_correct_detections: 1
plot_time_point_metrics: 1
make_quality_metrics: 1 
```


## License

This package is licensed under [PolyForm Noncommercial 1.0.0 license](https://polyformproject.org/wp-content/uploads/2020/05/PolyForm-Noncommercial-1.0.0.txt).


