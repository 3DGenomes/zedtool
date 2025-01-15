#!/bin/sh

# This script converts a SRX file to a picasso file by changing some column names
# and adding the frame column.

usage() {
    echo "Usage: $0 <filename>"
    echo "Converts a SRX file to a picasso file. The SRX file must have the following columns:"
    echo "frame, image-ID, x, y, z, photon-count, background11, precisionx, precisiony, precisionz"
    echo "The input file has a .csv extension and the output file has a .hd5 extension."
    exit 1
}

# frame -> srx_frame
# image-ID -> frame
# x -> x_nm
# y -> y_nm
# z -> z_nm
# photon-count -> intensity_photon
# background11 -> offset_photon
# precisionx -> sigma1_nm
# precisiony -> sigma2_nm
# precisionz -> uncertainty_xy_nm


# Check if the input file exists
if [ ! -f "$1.csv" ]; then
    echo "Error: $1.csv does not exist."
    usage
fi

tmpfile=$1.tsv

head -n 1 $1.csv | \
    sed 's/,frame,/,srx_frame,/' | \
    sed 's/image-ID,/frame,/' | \
    sed 's/,x,/,x_nm,/' | \
    sed 's/,y,/,y_nm,/' | \
    sed 's/,z,/,z_nm,/' | \
    sed 's/,photon-count,/,intensity_photon,/' | \
    sed 's/,background11,/,offset_photon,/' | \
    sed 's/,precisionx,/,sigma1_nm,/' | \
    sed 's/,precisiony,/,sigma2_nm,/' | \
    sed 's/,precisionz,/,uncertainty_xy_nm,/' > $tmpfile

grep -v frame $1.csv >> $tmpfile
picasso csv2hdf $tmpfile 1
# rm tmp.csv
