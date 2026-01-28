# Manual

This file describes how to do QC assessment and artefact correction to SMLM data using zedtool

## Workflow overview
### Steps

<img src="images/flowchart.png" alt="Flowchart of workflow." width="200"/>

The zedtool processing pipeline (see the diagram above) follows these main steps:

1. Pre-process
   - Merge experiments and apply any pre-made drift corrections.
   - Filter detections and locate candidate fiducial markers.
   - Filter fiducials by quality metrics.

2. Plots \(Pre-correction\)
   - Visualize detections and segmented fiducials 
   - Plot detections and fiducial quality metrics to assess data before corrections.
   - Plot drift and movement diagnostics.

3. Corrections
   - Rotation correction of detections.
   - Drift correction using fitted fiducials.
   - \(\Delta\)z correction for axial offsets.

4. Plots \(Post-correction\)
   - Re-do pre-correction plots to assess the effect of corrections.

5. Outputs
   - Export corrected detection files, labeled detections and details of corrections. 

### Files

<img src="images/files.png" alt="Summary of input and output files." width="600"/>

#### Inputs
- `config.yaml` — pipeline configuration (sets `output_dir`, fitting/filtering options, multiprocessing, column names).
- `detections_file.csv` — localisations assumed to be in a csv file.
- `concatenate_detections_file.csv` — optional extra localisations used when joining multiple experiments.
- `concatenate_offset_file.csv` — numeric offsets applied when concatenating experiments. 
Columns are those that need to be offset (e.g., `x`, `y`, `z`,`time-point`, `image-ID`) and the row contains 
the value that has to be added to the second experiment to make it consistent with the first.

#### Uncorrected outputs
- `output_dir` — top-level output directory (set in `config.yaml`). Contains binned detection files,
drift correction details, logfile and backup of config file.
- `output_dir/summary_plots` — overview plots and QC figures.
- `output_dir/time_point_metrics` — per-time-point quality metrics and tables.
- `output_dir/fiducial_step_fit` — intermediate fit results for fiducial fitting steps generating drift corrections.
- `output_dir/fiducials` — per-fiducial diagnostics and data files.
- `output_dir/fiducials/histograms` — histograms and distribution plots for fiducial metrics.
- `output_dir/fiducials/f_idnum_z_value_y_value_x_value` — per-fiducial files (naming encodes fiducial id and mean coordinates).

#### Corrected outputs
- `output_dir_corrected` — mirror of `output_dir` containing corrected detection tables and plots.
- Subfolders under `output_dir_corrected` replicate the same structure: `summary_plots`, `time_point_metrics`, `fiducials`, `fiducials/histograms`, and per-fiducial files named as above.

### Localisation file format

A localisation file is a CSV file where each row represents a single localisation. The column names are specified in the config file, 
with default names listed below. Additional columns are allowed and will be preserved. 
If a column is marked as `zero` in the config file then it need not be present in the localisation file and will be created and initialised to `0`.

#### Default Column Names
The default column names follow the Vutara format, written here in yaml-style as they would appear in the config file.

```yaml
frame_col: frame           # number of images since the start of the time-point
image_id_col: image-ID     # number of images since the start of the experiment
z_step_col: z-step         # z position of the imaging plane encoded as an integer
cycle_col: cycle           # number of sweeps of the objective since the start of the time-point
time_point_col: time-point # the number of sequential labeling rounds since the start of the experiment 
x_col: x
y_col: y
z_col: z
x_sd_col: precisionx
y_sd_col: precisiony
z_sd_col: precisionz
photons_col: photon-count
chisq_col: chisq
deltaz_col: deltaz         # distance between z and imaging plane - automatically created if missing
log_likelihood_col: log-likelihood
llr_col: llr
probe_col: vis-probe
```

##### Sample localisation file Vutara format

```csv
frame,image-ID,z-step,cycle,time-point,x,y,z,precisionx,precisiony,precisionz,photon-count,chisq,deltaz,log-likelihood,llr,vis-probe,area
1,img001,0,1,1,1234.56,2345.67,150.2,10.5,11.2,20.0,1500,1.02,0.0,-345.6,12.3,A,45.2
2,img001,0,1,1,1235.10,2346.00,149.8,10.8,11.0,19.5,1400,0.98,0.0,-344.8,11.7,A,48.1
```

#### Specifying column names

The following is a portion of a config file for processing a file made
using Thunderstorm:

```yaml
image_id_col: frame 
frame_col: frame
time_point_col: zeros
x_col: "x [nm]"
y_col: "y [nm]"
z_col: zeros
x_sd_col: "sigma [nm]"
y_sd_col: "sigma [nm]"
z_sd_col: zeros
chisq_col: zeros
photons_col: "intensity [photon]"
z_step_col: zeros
cycle_col: zeros
deltaz_col: deltaz # distance between z and imaging plane - automatically created if missing
log_likelihood_col: zeros
llr_col: zeros
probe_col: zeros
```

##### Sample localisations in Thunderstorm format:

```csv
"id","frame","x [nm]","y [nm]","sigma [nm]","intensity [photon]","offset [photon]","bkgstd [photon]","uncertainty [nm]"
1.0,1.0,2127.3318498388653,16340.527461169107,107.22122353737244,264.91639465558524,14.438872207683776,3.8321667129345767,10.931820904967148
2.0,1.0,2806.4236907180607,14057.775120679511,141.50261917701098,1393.176002510907,17.689110385550034,5.767525524125411,5.753838400571674
3.0,1.0,2761.5194198808767,15576.599860844663,111.24096252413386,690.124940865871,19.538992838523296,5.8301625537768205,6.926179611769085
4.0,1.0,3741.604997974105,9707.33175717059,112.73968099245164,428.6732078212021,13.790800037666964,3.801471070224385,8.067621278661006
```

## Quality assessment

### Plots

![Plot Example](images/fiducials_plot.png)

### Plots Produced by Zedtool


- `output_dir/summary_plots`: Pre-correction overview plots.
- `output_dir/time_point_metrics`: Drift and movement diagnostics.
- `output_dir/fiducials`: Fiducial diagnostics and histograms.
- `output_dir_corrected/summary_plots`: Post-correction overview plots.
- `output_dir/fiducials/f_idnum_z_value_y_value_x_value`: Per-fiducial plots.

These plots are essential for ensuring the accuracy of corrections and the reliability of the processed data.


### Tables

Text

### Fiducials

Text

### Quality filtering without correction

## Adjusting segmentation settings

```bash
setting: value
```

## Quality filtering

Text

```bash
setting: value
setting: value

```

## Localisation filtering

Text

```bash
setting: value
setting: value
```

## Density filtering

Text

```bash
setting: value
setting: value
```

## Fiducial markers

Text

### Segmenting

### Assessing

### Filtering and plotting

* quantile or sd_outlier_cutoff
* fiducial filtering quantities: 'n_detections', 'n_images', 'detections_per_image', 'x_mean', 'x_sd', 'y_mean', 'y_sd', 'z_mean', 'z_sd', 'photons_mean', 'photons_sd', 'area', 'x_madr', 'y_madr', 'z_madr'
* plotting covariates: 'image_id_col', 'z_step_col', 'cycle_col', 'time_point_col', 'deltaz_col', 'photons_col', 'x_sd_col', 'y_sd_col', 'z_sd_col'
* detections_per_image also has cutoff from max_detections_per_image

### Selection

## Rotation correction

* How the method works
* When to use it

## Drift correction

* How the method works

### Fitting parameters

### Choosing fiducials

### Quality metrics

### Importing and exporting drift correction

## Deltaz correction

* How the method works
* When to use it

## Joining experiments

* Files required and offsets

## Further data exploration

### Telling between drift and localisation error

### Movement at small time scales

## Example of typical use cases

* Example 1: Simple QC and drift correction
* Example 2: Experiment in two parts that require joining
* Example 3: Experiment requiring manual fiducial selection
