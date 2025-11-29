# Manual
Performing QC and artifact correction with zedtool

## Workflow overview
Text

### Steps

<!-- ![Plot Example](images/flowchart.png) --> 
<img src="images/flowchart.png" alt="Plot Example" width="300"/>

### Input and output files
![Plot Example](images/files.png)

### Detection file format

#### Specifying column names
* Adding and using the zeros column for use in missing columns.
 
## Quality assessment

### Plots 

![Plot Example](images/fiducials_plot.png)

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

## Position filtering 
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

### Filtering amd plotting
* quantile or sd_outlier_cutoff
* filtering quantities: 'n_detections', 'n_images', 'detections_per_image', 'x_mean', 'x_sd', 'y_mean', 'y_sd', 'z_mean', 'z_sd', 'photons_mean', 'photons_sd', 'area', 'x_madr', 'y_madr', 'z_madr'
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

## Non-Vutara file formats
* Example Thunderstorm dataset
* Use of "zeros" column for missing columns

## Further data exploration

### Telling between drift and localisation error

### Movement at small time scales

## Example of typical use cases
* Example 1: Simple QC and drift correction
* Example 2: Experiment in two parts that require joining
* Example 3: Experiment requiring manual fiducial selection


