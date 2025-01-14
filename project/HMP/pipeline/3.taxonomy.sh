#!/bin/bash

# Taxonomy

# Step 1.
qiime feature-classifier classify-sklearn \
  --i-classifier silva-138-99-nb-classifier.qza \
  --i-reads rep-seqs.qza \ 
  --o-classification taxonomy.qza

# Step 2.
qiime metadata tabulate \
  --m-input-file taxonomy.qza \
  --o-visualization taxonomy.qzv

# Step 3.
qiime taxa barplot \
  --i-table table.qza \ 
  --i-taxonomy taxonomy.qza \
  --m-metadata-file metadata.tsv \
  --o-visualization taxa-bar-plots.qzv

