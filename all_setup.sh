wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UgyMGS0rPHjdqmS-nsp0xuwiU-IA1n5B' -O data.zip
unzip data.zip
bash
conda env create -f conda/environment.yml
conda activate roc_env
conda install jupyterlab
