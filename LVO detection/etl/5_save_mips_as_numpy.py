import pandas as pd
import numpy as np
import pydicom
import glob
import os, os.path as osp

from sklearn.model_selection import KFold, StratifiedKFold
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm


def create_double_cv(df, id_col, outer_splits, inner_splits, stratified=None, seed=88):
    df['outer'] = 888
    splitter = KFold if stratified is None else StratifiedKFold
    outer_spl = splitter(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_counter = 0
    for outer_train, outer_test in outer_spl.split(df) if stratified is None else outer_spl.split(df, df[stratified]):
        df.loc[outer_test, 'outer'] = outer_counter
        inner_spl = splitter(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_counter = 0
        df['inner{}'.format(outer_counter)] = 888
        inner_df = df[df['outer'] != outer_counter].reset_index(drop=True)
        # Determine which IDs should be assigned to inner train
        for inner_train, inner_valid in inner_spl.split(inner_df) if stratified is None else inner_spl.split(inner_df, inner_df[stratified]):
            inner_train_ids = inner_df.loc[inner_valid, id_col]
            df.loc[df[id_col].isin(inner_train_ids), 'inner{}'.format(outer_counter)] = inner_counter
            inner_counter += 1
        outer_counter += 1
    return df


def load_dicom_stack(folder):
    files = glob.glob(osp.join(folder, '*.dcm'))
    loaded = [pydicom.dcmread(f) for f in files]
    z_positions = np.argsort([float(l.ImagePositionPatient[2]) for l in loaded])
    array = np.asarray([l.pixel_array for l in loaded])
    array = array[z_positions]
    array = array * float(loaded[0].RescaleSlope)
    array = array + float(loaded[0].RescaleIntercept)
    return array.astype('int16')


def resample(array):
    assert array.ndim == 3
    if len(array) > 64:
        array = zoom(array, [64./array.shape[0], 1, 1], order=0, prefilter=False)
    arr = zoom(array, [1, 0.5, 0.5], order=1, prefilter=False)
    return arr


SAVEDIR = '../../data/rih/mips/'

df = pd.read_csv('../../data/rih/quick_arterial_mips.csv')

failures = []
for seri, acch in tqdm(zip(df['series'], df['AccessionNumber']), total=len(df)):
    try:
        array = load_dicom_stack(seri)
        array = resample(array)
        acch_dir = osp.join(SAVEDIR, acch)
        if not osp.exists(acch_dir):
            os.makedirs(acch_dir)
        np.save(osp.join(acch_dir, seri.split('/')[-1]+'.npy'), array)
    except Exception as e:
        print('{} failed : {} !'.format(seri, e))
        failures.append(seri)
        continue

# Assign splits
df['pid'] = [_.split('/')[5] for _ in df['series']]
pids = df[['pid']].drop_duplicates().reset_index(drop=True)
#pids = create_double_cv(pids, 'pid', 8, 12)
pids = create_double_cv(pids, 'pid', 10, 10)
df = df.merge(pids, on='pid')

# Generate filenames
df['imgfile'] = [osp.join(acch, seri.split('/')[-1]+'.npy') for acch, seri in zip(df['AccessionNumber'], df['series'])]

df.to_csv('../../data/rih/train_quick_mips_with_splits_cv10x10.csv', index=False)










