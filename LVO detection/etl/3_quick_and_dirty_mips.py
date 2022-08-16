import pandas as pd
import hashlib
import re


def re_mip(s): return bool(re.search('MIP', s.upper()))


def re_arterial(s): return bool(re.search('ART', s.upper()))


def re_venous(s): return bool(re.search('VEN', s.upper()))


metadata_df = pd.read_csv('../../data/rih/series_metadata.csv')
series_desc = pd.read_csv('series_description.csv')
series_info = pd.read_csv('../../data/rih/series.csv')
metadata_df = metadata_df.merge(series_info, on=['AccessionNumber', 'SeriesNumber'])
metadata_df['SeriesDescription'] = [_.upper().replace(' ', '_') for _ in metadata_df['SeriesDescription']]

mips_df = metadata_df[metadata_df['SeriesDescription'].apply(re_mip)]
mips_df.loc[:,'RoundedSliceThickness'] = mips_df['SliceThickness'].apply(round)
mips_df = mips_df[((mips_df['RoundedSliceThickness'] == 24) | (mips_df['SeriesDescription'].apply(re_arterial)))]

mips_list = [_[1] for _ in mips_df.groupby('AccessionNumber')]

arterial_mips_list = []
for _df in mips_list:
    _arterial_df = _df[~_df['SeriesDescription'].apply(re_venous)]
    if len(_arterial_df) > 1:
        if len(_arterial_df[['num_files', 'PixelSpacing', 'SliceThickness', 'SeriesDescription']].drop_duplicates()) == 1:
            _arterial_df = _arterial_df.iloc[0:1]
    arterial_mips_list.append(_arterial_df)

multiple_rows = [_ for _ in arterial_mips_list if len(_) > 1]
deduped = []
for mr in multiple_rows:
    # First, check if one of them has arterial in description name
    tmp_mr = mr[mr['SeriesDescription'].apply(re_arterial)]
    if len(tmp_mr) == 1:
        deduped.append(tmp_mr)
        continue
    # If not, then prioritize:
    #   1. Highest # of slices
    #   2. Smaller pixel spacing
    mr = mr.sort_values(['num_files', 'PixelSpacing'], ascending=[False, True])
    deduped.append(mr.iloc[0:1])

arterial_mips_list.extend(deduped)
arterial_mips = pd.concat(arterial_mips_list)

# Labels
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

arterial_mips = arterial_mips.merge(labels_df[['AccessionNumber','label']], on='AccessionNumber')
desired_series = [
    'THICK_CAROTID_BRAIN_CTA_MIP',
    'ARTERIAL_THICK_AXIAL_MIP',
    'THICK_AXIAL_NECK_BRAIN_CTA_MIP',
    '24X4_AXIAL_HEAD_AND_NECK_MIP',
    'THICK_AXIAL_BRAIN_CAROTID_MIP',
    '24X4_AXIAL_MIP'
]
arterial_mips = arterial_mips[arterial_mips['SeriesDescription'].isin(desired_series)]
arterial_mips['label'].value_counts()              
arterial_mips.to_csv('../../data/rih/quick_arterial_mips.csv', index=False)




