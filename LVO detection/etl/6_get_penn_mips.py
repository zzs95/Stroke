import pydicom
import pickle
import pandas as pd
import numpy as np
import glob, os, os.path as osp

from tqdm import tqdm
from scipy.ndimage.interpolation import zoom


def determine_plane(o):
    o = [round(x) for x in o]
    plane = np.cross(o[0:3], o[3:6])
    plane = [abs(x) for x in plane]
    return np.argmax(plane)


def generate_mip(arr, spacing, thickness=24, skip=4):
    thickness = int(np.round(thickness / spacing))
    skip = int(np.round(skip / spacing))
    mip_slices = []
    for i in range(0, len(arr)-skip, skip):
        mip_slices.append(np.max(arr[i:i+thickness], axis=0))
    return np.asarray(mip_slices[:-2])


def generate_mip_from_dicoms(folder):
    files = glob.glob(osp.join(folder, '*'))
    loaded = np.asarray([pydicom.dcmread(f) for f in files])
    # Get rid of DICOMs with different array shapes from majority
    array_shapes = np.asarray([l.pixel_array.shape for l in loaded])
    shapes, counts = np.unique(array_shapes, axis=0, return_counts=True)
    shape = shapes[np.argmax(counts)]
    keep = np.where((array_shapes == shape).mean(axis=1) == 1)[0]
    loaded = loaded[keep]
    # Get rid of DICOMs with different ImageType from majority
    image_types = np.asarray(['_'.join(l.ImageType) for l in loaded])
    imtypes, counts = np.unique(image_types, return_counts=True)
    imtype = imtypes[np.argmax(counts)]
    keep = np.where((image_types == imtype))[0]
    loaded = loaded[keep]
    # Determine plane
    pos = determine_plane(loaded[0].ImageOrientationPatient)
    positions = np.asarray([l.ImagePositionPatient for l in loaded]).astype('float32')
    instances = [float(l.InstanceNumber) for l in loaded]
    positions = positions[:,pos]
    array = np.asarray([l.pixel_array for l in loaded])
    array = array[np.argsort(positions)]
    positions = np.sort(positions)
    slice_spacing = positions[1:] - positions[:-1]
    slice_spacing = np.median(np.abs(slice_spacing))    
    array = array * float(loaded[0].RescaleSlope)
    array = array + float(loaded[0].RescaleIntercept)
    # Generate MIP
    array = generate_mip(array, slice_spacing, 24,4)
    return array.astype('int16')


def resample(array):
    assert array.ndim == 3
    if len(array) > 64:
        array = zoom(array, [64./array.shape[0], 1, 1], order=0, prefilter=False)
    arr = zoom(array, [1, 0.5, 0.5], order=1, prefilter=False)
    return arr


SAVE_MIPS_DIR = '../../data/penn/mips-numpy/'
if not osp.exists(SAVE_MIPS_DIR): os.makedirs(SAVE_MIPS_DIR)


with open('../../data/penn/predictions.pkl', 'rb') as f:
    series_preds = pickle.load(f)

y_pred = series_preds['y_pred']
series = series_preds['series']
assert len(y_pred) == len(series)

keep = [i for i, _ in enumerate(y_pred) if type(_) != type(None)]
y_pred = np.vstack(np.asarray(y_pred)[keep])
series = np.asarray(series)[keep]
accessions = [int(_.split('/')[2].split('_')[0]) for _ in series]

MIP, THIN = 0, 4

df = pd.DataFrame({
        'accession': accessions,
        'series': series,
        'mip_prob': y_pred[:,MIP],
        'thin_prob': y_pred[:,THIN]
    })

df_list = [_ for k,_ in df.groupby('accession')]
# Looks like we need to get the thins and generate MIPs ...

for each_df in tqdm(df_list, total=len(df_list)):
    each_df = each_df.sort_values('thin_prob', ascending=False)
    folder = osp.join('../../data/', str(each_df['series'].iloc[0]))
    mip = generate_mip_from_dicoms(folder)
    mip = resample(mip)
    acch_dir = osp.join(SAVE_MIPS_DIR, str(each_df['accession'].iloc[0]))
    if not osp.exists(acch_dir): os.makedirs(acch_dir)
    np.save(osp.join(acch_dir, each_df['series'].iloc[0].split('/')[-1]+'.npy'), mip)




