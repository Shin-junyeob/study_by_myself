# Setup

### Install the Qiime2 enviroment

![](https://github.com/qiime2/qiime2/workflows/ci-dev/badge.svg)

[Qiime2 Website]()

```bash
user@:~$ conda update conda
```

```bash
user@:~$ conda env create -n qiime2-amplicon-2024.5 --file https://data.qiime2.org/distro/amplicon/qiime2-amplicon-2024.5-py39-linux-conda.yml
```

```bash
user@:~$ conda env create -n qiime2-metagenome-2024.5 --file https://data.qiime2.org/distro/metagenome/qiime2-metagenome-2024.5-py39-linux-conda.yml
```    

```bash
user@:~$ conda env create -n qiime2-tiny-2024.5 --file https://data.qiime2.org/distro/tiny/qiime2-tiny-2024.5-py39-linux-conda.yml
```


### Get data
Download and prepare the datasets (HMP Data Portal, SILVA REFERENCE) via the follwing link: \
    [Amplicon](https://portal.hmpdacc.org/search/s?facetTab=files&filters=%7B%22op%22:%22and%22,%22content%22:%5B%7B%22op%22:%22in%22,%22content%22:%7B%22field%22:%22subject.project_name%22,%22value%22:%5B%22Human%20Microbiome%20Project%20(HMP)%22%5D%7D%7D,%7B%22op%22:%22in%22,%22content%22:%7B%22field%22:%22sample.body_site%22,%22value%22:%5B%22gastrointestinal%20tract%22%5D%7D%7D,%7B%22op%22:%22in%22,%22content%22:%7B%22field%22:%22file.node_type%22,%22value%22:%5B%2216s_raw_seq_set%22%5D%7D%7D%5D%7D) \
    [Metagenome](https://portal.hmpdacc.org/search/s?facetTab=files&filters=%7B%22op%22:%22and%22,%22content%22:%5B%7B%22op%22:%22in%22,%22content%22:%7B%22field%22:%22subject.project_name%22,%22value%22:%5B%22Human%20Microbiome%20Project%20(HMP)%22%5D%7D%7D,%7B%22op%22:%22in%22,%22content%22:%7B%22field%22:%22sample.body_site%22,%22value%22:%5B%22gastrointestinal%20tract%22%5D%7D%7D,%7B%22op%22:%22in%22,%22content%22:%7B%22field%22:%22file.node_type%22,%22value%22:%5B%22wgs_raw_seq_set%22%5D%7D%7D%5D%7D) \
    [Reference](https://www.arb-silva.de/download/archive/)

```bash
user@:~$ bash get_datasets.sh
```
```shell
#!/bin/bash

############################
#   Amplicon dataset
wget https://downloads.hmpdacc.org/data/HMR16S/HMDEMO/SRP002468/SRR050905.sff.bz2 https://downloads.hmpdacc.org/data/HMR16S/HMDEMO/SRP002468/SRR050909.sff.bz2 https://downloads.hmpdacc.org/data/HMR16S/HMDEMO/SRP002468/SRR050929.sff.bz2 https://downloads.hmpdacc.org/data/HMR16S/HMDEMO/SRP002468/SRR050933.sff.bz2
############################

############################
#   Metagenome dataset
wget https://downloads.hmpdacc.org/data/Illumina/HMDEMO/SRP002468/affected/SRS267354.tar.bz2 https://downloads.hmpdacc.org/data/Illumina/HMDEMO/SRP002468/affected/SRS267355.tar.bz2 https://downloads.hmpdacc.org/data/Illumina/HMDEMO/SRP002468/affected/SRS260328.tar.bz2 https://downloads.hmpdacc.org/data/Illumina/HMDEMO/SRP002468/affected/SRS260329.tar.bz2
############################

############################
#   SILVA dataset
wget https://data.qiime2.org/2024.5/common/silva-138-99-seqs.qza https://data.qiime2.org/2024.5/common/silva-138-99-tax.qza https://data.qiime2.org/classifiers/sklearn-1.4.2/silva/silva-138-99-nb-classifier.qza
############################
```
### Transformation
preprocess data into a format compatible with Qiime2.

```bash
user@:~$ ./preprocess.sh
```
```shell
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
```

# Qiime2 Distribution
Using built-in Qiime2 functionalities for analysis