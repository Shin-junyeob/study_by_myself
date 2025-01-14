#!/bin/bash

# Phylogeny

## align
qiime phylogeny align-to-tree-mafft-fasttree \
  --i-sequences rep-seqs.qza \
  --o-alignment aligned-rep-seqs.qza \
  --o-masked-alignment masked-aligned-rep-seqs.qza \
  --o-tree unrooted-tree.qza \
  --o-rooted-tree rooted-tree.qza

## core-metrics-results
qiime diversity core-metrics-phylogenetic \
  --i-phylogeny rooted-tree.qza \
  --i-table table.qza \
  --p-sampling-depth 2915 \
  --m-metadata-file metadata.tsv  \
  --output-dir core-metrics-results

###table.qzv에서 가장 적은 sequence를 가지고 있는 sample의 sequence 수를 확인 seq_130이 2915개 가지고있음

## sampling-depth check

qiime diversity core-metrics-phylogenetic \
  --i-phylogeny rooted-tree.qza \
  --i-table table.qza \
  --p-sampling-depth 1000 \
  --m-metadata-file metadata.tsv  \
  --output-dir ccc/core-metrics-results
