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


def determine_plane(o):
    o = [round(x) for x in o]
    plane = np.cross(o[0:3], o[3:6])
    plane = [abs(x) for x in plane]
    return np.argmax(plane)


def cov(x): return np.std(x) / np.abs(np.mean(x))


def load_dicom_stack(folder, mismatch='position'):
    files = glob.glob(osp.join(folder, '*'))
    loaded = np.asarray([pydicom.dcmread(f) for f in files])
    # Get rid of DICOMs with different array shapes from majority
    array_shapes = [l.pixel_array.shape for l in loaded]
    array_shapes = np.asarray([','.join([str(_) for _ in arr_shap]) for arr_shap in array_shapes])
    shapes, counts = np.unique(array_shapes, return_counts=True)
    shape = shapes[np.argmax(counts)]
    keep = np.where(array_shapes == shape)[0]
    loaded = loaded[keep]
    # Get rid of DICOMs with different ImageType from majority
    image_types = np.asarray(['_'.join(l.ImageType) for l in loaded])
    imtypes, counts = np.unique(image_types, return_counts=True)
    imtype = imtypes[np.argmax(counts)]
    keep = np.where((image_types == imtype))[0]
    loaded = loaded[keep]
    # Determine plane
    #pos = determine_plane(loaded[0].ImageOrientationPatient)
    positions = np.asarray([l.ImagePositionPatient for l in loaded]).astype('float32')
    instances = [float(l.InstanceNumber) for l in loaded]
    cov_pos = np.hstack([cov(positions[:,i]) for i in range(positions.shape[-1])])
    pos = np.argmax(cov_pos)
    positions = positions[:,pos]
    if np.mean(np.argsort(positions) == np.argsort(instances)) != 1:
        # Could be reversed
        if np.mean(np.argsort(positions)[::-1] == np.argsort(instances)) != 1:
            print('Mismatch between `ImagePositionPatient` and `InstanceNumber` sorting !')
            if mismatch == 'instance':
                print('Using `InstanceNumber` ...')
                positions = instances
            elif mismatch == 'position':
                print('Using `ImagePositionPatient` ...')
    array = np.asarray([l.pixel_array for l in loaded])
    array = array[np.argsort(positions)]
    array = array * float(loaded[0].RescaleSlope)
    array = array + float(loaded[0].RescaleIntercept)
    return array.astype('int16')


def resample(array):
    assert array.ndim == 3
    if len(array) > 64:
        array = zoom(array, [64./array.shape[0], 1, 1], order=0, prefilter=False)
    arr = zoom(array, [1, 0.5, 0.5], order=1, prefilter=False)
    return arr


SAVEDIR = '../../data/penn/numpy/'

metadata_df = pd.read_csv('../../data/penn/series_metadata.csv')

failures = []
for seri, acch in tqdm(zip(metadata_df['series'], metadata_df['AccessionNumber']), total=len(metadata_df)):
    try:
        array = load_dicom_stack(osp.join('../../data/', seri))
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
sampled_df['pid'] = [_.split('/')[5] for _ in sampled_df['series']]
pids = sampled_df[['pid']].drop_duplicates().reset_index(drop=True)
pids = create_double_cv(pids, 'pid', 10, 10)
sampled_df = sampled_df.merge(pids, on='pid')

# Generate filenames
sampled_df['imgfile'] = [osp.join(acch, seri.split('/')[-1]+'.npy') for acch, seri in zip(sampled_df['AccessionNumber'], sampled_df['series'])]

sampled_df.to_csv('../../data/rih/train_series_sample_with_splits.csv', index=False)










