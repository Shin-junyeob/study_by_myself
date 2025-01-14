#!/bin/bash

bzip2 -dk *.sff.bz2

conda activate qiime2-amplicon-2024.5

############################
#   transform each .sff files to .qza files
for file in *.sff; do
    qiime tools import \
        --type 'SampleData[SequencesWithQuality]' \
        --input-path "$file" \
        --output-path "${file%.sff}.qza"
done
############################


############################
#   transform all .sff files to .qza
output_file="manifest.csv"
echo "sample-id,absolute-filepath" > $output_file

for file in *.sff; do
    sample_id=$(basename "$file" .sff)
    abs_path=$(realpath "$file")
    echo "$sample_id,$abs_path" >> $output_file
done

qiime tools import \
--type 'SampleData[SequencesWithQuality]' \
--input-path manifest.csv \
--output-path combined.qza \
--input-format SingleEndFastqManifestPhred33V2
############################