#!/bin/bash

# Demultiplexing

qiime tools import \
  --type 'SampleData[SequencesWithQuality]' \
  --input-format SingleEndFastqManifestPhred33V2 \
  --input-path manifest.tsv \
  --output-path demux.qza

qiime demux summarize \
  --i-data demux.qza \
  --o-visualization demux.qzv

