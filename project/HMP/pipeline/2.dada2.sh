#!/bin/bash

# DADA2

qiime dada2 denoise-single \
  --i-demultiplexed-seqs demux.qza \
  --p-trunc-len 300 \
  --p-n-threads 0 \
  --o-table table-dada2.qza \
  --o-representative-sequences rep-seqs-dada2.qza \
  --o-denoising-stats stats-dada2.qza

# Check for sequence 
cp rep-seqs-dada2.qza rep-seqs.qza 

qiime tools export --input-path rep-seqs.qza  --output-path rep-seqs

less -S  dna-sequences.fasta # ASV(fasta format) = dna-sequences.fasta

# Check for feature count
grep '>' -c dna-sequences.fasta

# Use for any other analysis
cp table-dada2.qza table.qza

