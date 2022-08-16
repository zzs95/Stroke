conda create -y -n pytorch_p37_t14 python=3.7 pip
conda activate pytorch_p37_t14
conda install -y pytorch torchvision -c pytorch
conda install -y -c conda-forge opencv
conda install -y scikit-learn scikit-image pandas pyyaml tqdm
conda install -y -c conda-forge gdcm

pip install pretrainedmodels albumentations kaggle pyarrow iterative-stratification
