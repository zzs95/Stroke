conda create -y -n pytorch_p37_nt python=3.7.7 pip
conda activate pytorch_p37_nt
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch-nightly
conda install -y -c conda-forge opencv
conda install -y scikit-learn scikit-image pandas pyyaml tqdm
conda install -y -c conda-forge gdcm

pip install pydicom pretrainedmodels albumentations kaggle pyarrow iterative-stratification

