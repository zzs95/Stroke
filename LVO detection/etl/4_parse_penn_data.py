import pydicom
import pandas as pd
import numpy as np
import glob
import os, os.path as osp
import re

from collections import defaultdict
from tqdm import tqdm


def grab(dcm, attr):
    try:
        return getattr(dcm, attr)
    except AttributeError:
        return None


dicom_files = []
penn_dir = list(os.walk('../../data/penn/'))
for root, dirs, files in tqdm(penn_dir, total=len(penn_dir)):
    for fi in files:
        filename = osp.join(root, fi)
        if osp.isfile(filename):
            dicom_files.append(filename)

dicom_files = [_ for _ in dicom_files if len(_.split('/')[-1].split('.')) == 1]
dicom_files = [_ for _ in dicom_files if _.split('/')[-1] != 'DICOMDIR']

studies = ['/'.join(_.split('/')[:6]) for _ in dicom_files]
series = ['/'.join(_.split('/')[:-1]) for _ in dicom_files]

dcm_df = pd.DataFrame({
        'study': studies, 'series': series, 'file': dicom_files
    })

metadata_dict = defaultdict(list)

for seri, seri_df in tqdm(dcm_df.groupby('series'), total=len(dcm_df['series'].unique())):
    try:
        if len(seri_df) < 5: continue
        tmp_dcm = pydicom.dcmread(seri_df['file'].iloc[0], stop_before_pixels=True)
        metadata_dict['series'].append(seri)
        metadata_dict['num_files'].append(len(seri_df))
        metadata_dict['PixelSpacing'].append(grab(tmp_dcm, 'PixelSpacing'))
        metadata_dict['SliceThickness'].append(grab(tmp_dcm, 'SliceThickness'))
        metadata_dict['SeriesNumber'].append(grab(tmp_dcm, 'SeriesNumber'))
        metadata_dict['ImageType'].append(grab(tmp_dcm, 'ImageType'))
    except Exception as e:
        print(e)
        continue

metadata_dict['PixelSpacing'] = [float(_[0]) if type(_) != type(None) else _ for _ in metadata_dict['PixelSpacing']]
metadata_dict['SliceThickness'] = [float(_) if type(_) != type(None) else _ for _ in metadata_dict['SliceThickness']]
metadata_dict['SeriesNumber'] = [int(_) if type(_) != type(None) else _ for _ in metadata_dict['SeriesNumber']]
metadata_dict['ImageType'] = ['_'.join(_) if type(_) != type(None) else _ for _ in metadata_dict['ImageType']]

metadata_df = pd.DataFrame(metadata_dict)
metadata_df['AccessionNumber'] = [_.split('/')[5] for _ in metadata_df['series']]
metadata_df['series'] = [_.replace('../../data/', '') for _ in metadata_df['series']]

metadata_df.to_csv('../../data/penn/series_metadata.csv', index=False)
