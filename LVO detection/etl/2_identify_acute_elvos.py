import pandas as pd
import numpy as np
import hashlib
import glob, os, os.path as osp


def get_slice_spacing(folder):
    files = glob.glob(osp.join(folder, '*.dcm'))
    loaded = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    z_positions = np.sort([float(l.ImagePositionPatient[2]) for l in loaded])
    return np.median(z_positions[:-1]-z_positions[1:])


series_info = pd.read_csv('../../data/rih/series.csv')
metadata_df = pd.read_csv('../../data/rih/series_metadata.csv')
labels_df = pd.read_csv('../../data/rih/search_ELVO.csv')
labels_df = labels_df[['Accession Number', 'occlusion or not', 'Acute or chronic', ' vessels', 'Patient MRN']]
labels_df['AccessionNumber'] = [hashlib.md5(str(_).encode('utf-8')).hexdigest() for _ in labels_df['Accession Number']]
label_mapping = {
    'not': 0,
    'occlusion': 1,
    'occlusion<50%': 0,
    'no report': None
}
labels_df['label'] = [label_mapping[_] for _ in labels_df['occlusion or not']]
positive_label_mapping = {
    'chronic': 1,
    'acute': 2,
    'acute/chronic': 2,
    'weizhi': 0,
    'chronic/acute': 2,
    'chronic/actue': 2
}
labels_df.loc[labels_df['label'] == 1, 'label'] = [positive_label_mapping[_] for _ in labels_df.loc[labels_df['label'] == 1, 'Acute or chronic']]

df = metadata_df.merge(series_info, on=['AccessionNumber', 'SeriesNumber'])
df = df.merge(labels_df, on='AccessionNumber')
df['PatientMRN'] = [hashlib.md5(str(_).encode('utf-8')).hexdigest() for _ in df['Patient MRN']]
del df['Patient MRN']

occlusion = df[df['label'] > 0]
acute = df[df['label'] == 2]
acute = acute.sort_values('series').reset_index(drop=True)

# Some accession numbers are in labels CSV but missing from 
# series descriptions (series.csv)
missing_acutes = list(set(labels_df[labels_df['label'] == 2]['AccessionNumber']) - set(acute['AccessionNumber']))
print(len(acute['AccessionNumber'].unique()))
# 259->252
axial_mips = [
    '24X4 AXIAL HEAD AND NECK MIP', 
    '24X4 AXIAL MIP', 
    'ARTERIAL THICK AXIAL MIP',
    'THICK AXIAL NECK BRAIN CTA MIP', 
    'THICK CAROTID BRAIN CTA MIP', 
    'THICK CAROTID BRAIN CTA MIP 1', 
    'THICK CAROTID BRAIN CTA MIP 2'
]
acute_mips = acute[acute['SeriesDescription'].isin(axial_mips)].drop_duplicates().reset_index(drop=True)
len(acute_mips['AccessionNumber'].unique())
# 251
# Which one is missing...?
acute[~acute['AccessionNumber'].isin(list(acute_mips['AccessionNumber']))]
# The MIP we are looking for does not appear to be in here ...
# Deal with duplicates
dupes = pd.DataFrame(acute_mips['AccessionNumber'].value_counts()).reset_index()
dupes = list(dupes.loc[dupes['AccessionNumber'] > 1, 'index'])
#dupes = [_[1] for _ in acute_mips[acute_mips['AccessionNumber'].isin(dupes)].groupby('AccessionNumber')]
# Unclear which ones are correct ... So drop them
acute_mips = acute_mips[~acute_mips['AccessionNumber'].isin(dupes)]
# 241 ...
slice_spacing = []
from tqdm import tqdm
for seri in tqdm(acute_mips['series'], total=len(acute_mips)):
    slice_spacing.append(get_slice_spacing(seri))



