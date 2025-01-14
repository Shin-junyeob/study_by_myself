#!/bin/bash

# Alpha-group-significance

## Use core-metrics-results 

### alpha는 같은 group내의 다양성을 계산,  beta는 그룹 간의 다양성 분석

#### faith-pd는 계통 간의 거리를 반영하는 다양성을 측정 ###
#### Kruskal-Wallis - 3그룹 이상 차이를 보기 위한 분석
#### q-value에 집중.

qiime diversity alpha-group-significance \
  --i-alpha-diversity core-metrics-results/faith_pd_vector.qza \
  --m-metadata-file metadata.tsv \
  --o-visualization core-metrics-results/faith-pd-group-significance.qzv


#### 계통 간의 관계를 고려하는 다양성지수는 unweighted-unifrac distance

qiime diversity beta-group-significance \
  --i-distance-matrix core-metrics-results/unweighted_unifrac_distance_matrix.qza \
  --m-metadata-file -metadata.tsv \
  --m-metadata-column body-site \
  --o-visualization core-metrics-results/unweighted-unifrac-body-site-significance.qzv \
  --p-pairwise


qiime diversity beta-group-significance \
  --i-distance-matrix weighted_unifrac_distance_matrix.qza \
  --m-metadata-file metadata.tsv \
  --m-metadata-column body-site \
  --p-method anosim \
  --output-dir weighted/beta-sig