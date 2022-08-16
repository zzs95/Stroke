import pydicom
import pandas as pd
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


studies = glob.glob('../../data/rih/zips/*/*')
studies = [_ for _ in studies if _.split('/')[-1].split('.')[-1] != 'zip']

metadata_dict = defaultdict(list)

for study in tqdm(studies, total=len(studies)):
    series = glob.glob(osp.join(study, '*'))
    for seri in series:
        if re.search('CT', seri):
            try:
                dicoms = glob.glob(osp.join(seri, '*.dcm'))
                if len(dicoms) < 5: continue
                tmp_dcm = pydicom.dcmread(dicoms[0], stop_before_pixels=True)
                metadata_dict['series'].append(seri)
                metadata_dict['num_files'].append(len(dicoms))
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
metadata_df['AccessionNumber'] = [_.split('/')[-2].split()[0] for _ in metadata_df['series']]

metadata_df.to_csv('../../data/rih/series_metadata.csv', index=False)