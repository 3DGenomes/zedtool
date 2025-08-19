#!/usr/bin/env python3
import sys
import shutil
import os
import logging
from zedtool.process import read_config, read_detections, process_detections, pre_process_detections, post_process_detections
from zedtool import __version__
# Makes QC plots and metrics for an SMLM dataset.
# Optionally corrects drift and z-step related errors.
# This is a command line interface for the zedtool package.
def main(yaml_config_file: str) -> int:
    print_ascii_logo()
    print_version()

    config = read_config(yaml_config_file)

    if config is None:
        print(f"Failed to read {yaml_config_file}")
        return 1

    df = read_detections(config)
    if df is None or len(df) == 0:
        print(f"Failed to read and validate detections file {config['detections_file']}")
        return 1

    # get colnames for output file from input file of config
    if config['output_column_names'] is None:
        config['output_column_names'] = df.columns.tolist()
        # add image_ID to output column names to the front of the list if it is not already there
        # it gets created in pre_process_detections even if it is not in the input file
        if config['image_id_col'] is not None and config['image_id_col'] not in config['output_column_names']:
            config['output_column_names'].insert(0, config['image_id_col'])
    else:
        config['output_column_names'] = config['output_column_names'].split(',')
        # Check that the output column names are in the df
        for col in config['output_column_names']:
            if col not in df.columns:
                logging.warning(f"Column {col} specified in output_column_names not found in detections file")

    df = pre_process_detections(df, config)
    df, df_fiducials = process_detections(df, None, config)
    if df is None:
        print(f"Failed to process detections file {config['detections_file']}")
        return 1
    if config['making_corrrections']:
        df = post_process_detections(df, df_fiducials, config)
    if df is None:
        print(f"Failed to post process detections file {config['detections_file']}")
        return 1
    shutil.copy(yaml_config_file, os.path.join(config['output_dir'], 'config.yaml'))
    return 0

def print_version() -> str:
    ret = f"  Version: {__version__}\n"
    print(ret)
    return ret

def print_ascii_logo() -> str:
    ret = """
 +-+-+-+-+-+-+-+
 |z|e|d|t|o|o|l|
 +-+-+-+-+-+-+-+
     """
    print(ret)
    return ret

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: zedtool.py <config_file>")
        print_version()
        sys.exit(1)
    ret = main(sys.argv[1])
    print(ret)
    sys.exit(ret)
