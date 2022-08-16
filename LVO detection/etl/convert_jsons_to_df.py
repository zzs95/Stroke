import json
import glob
import pandas as pd
import os.path as osp
import hashlib

from tqdm import tqdm

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


ROOT = '/users/ipan/scratch/elvo-jsons/jsons/'

files = glob.glob(osp.join(ROOT, '*.json'))

df_list = []
for fi in tqdm(files, total=len(files)):
    js = load_json(fi)
    to_dict = {}
    for item in js:
        to_dict['AccessionNumber'] = [hashlib.md5(item['AccessionNumber'].encode('utf-8')).hexdigest()]
        to_dict['SeriesNumber'] = [item['SeriesNumber']]
        to_dict['SeriesDescription'] = [item['SeriesDescription']]
        df_list.append(pd.DataFrame(to_dict))


df = pd.concat(df_list)
df['SeriesDescription'] = [_.strip().upper() for _ in df['SeriesDescription']]
df.to_csv(osp.join(ROOT, '../series.csv'), index=False)

#

import pandas as pd

df0 = pd.read_csv('series.csv')
df0['AccessionNumber'] = [_[:16] for _ in df0['AccessionNumber']]
df1 = pd.read_csv('series_info.csv')

df = df0.merge(df1, left_on=['AccessionNumber','SeriesNumber'], right_on=['AccessionHash','SeriesNumber'])

mips = ['THICK CAROTID BRAIN CTA MIP', 'ARTERIAL THICK AXIAL MIP', 'THICK AXIAL NECK BRAIN CTA MIP', '24X4 AXIAL HEAD AND NECK MIP']
mip_df = df[df['SeriesDescription'].isin(mips)]
mip_df = mip_df[mip_df['SliceThickness'] > 20]

thins = ['1MM AXIAL BRAIN NECK CTA', 'THIN AXIAL NECK AND BRAIN CTA', 'CTA HEAD AND NECK', 'THIN AXIAL BRAIN NECK CTA',
         'AXIAL NECK AND BRAIN CTA', 'AXIAL NECK BRAIN CTA', 'AXIAL BRAIN NECK CTA', 'AXIAL CAROTID/BRAIN CTA', 'AXIAL BRAIN CAROTID CTA']
thin_df = df[df['SeriesDescription'].isin(thins)]

missing_df = df[~df['AccessionNumber'].isin(mip_df['AccessionNumber'])]