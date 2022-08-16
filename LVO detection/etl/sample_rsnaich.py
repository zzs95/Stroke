import pydicom
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

def load_dicom_stack(folder):
    files = glob.glob(folder+'/*.dcm')
    loaded = [pydicom.dcmread(f) for f in tqdm(files, total=len(files))]
    z_positions = [float(l.ImagePositionPatient)[2] for l in loaded]
    array = np.asarray([l.pixel_array for l in loaded])
    array = array[z_positions]
    return array, loaded[0].ImageType

ORIGINAL PRIMARY LOCALIZER CT_SOM5 TOP 
DERIVED PRIMARY AXIAL CT_SOM5 MIP 
DERIVED PRIMARY AXIAL CT_SOM5 MIP 
ORIGINAL PRIMARY LOCALIZER CT_SOM5 TOP
DERIVED PRIMARY AXIAL CT_SOM5 MPR 
DERIVED PRIMARY AXIAL CT_SOM5 MIP 
ORIGINAL PRIMARY AXIAL CT_SOM5 SPI 
DERIVED PRIMARY AXIAL CT_SOM5 MIP 
DERIVED PRIMARY AXIAL CT_SOM5 MIP 
ORIGINAL PRIMARY AXIAL CT_SOM5 SPI 
ORIGINAL PRIMARY AXIAL CT_SOM5 SPI


import pandas as pd
import numpy as np
import cv2
import os

from tqdm import tqdm 

NUMPYDIR = 'numpy-crop/train/'
NSAMPLES = 1000

df = pd.read_csv('stage_1_train_labels.csv')
df['sum'] = np.sum(df[['edh','sdh','iph','ivh','sah','any']], axis=1)

edh_sample = np.random.choice(df[(df['edh'] == 1) & (df['sum'] == 2)]['pid'], NSAMPLES, replace=False)
sdh_sample = np.random.choice(df[(df['sdh'] == 1) & (df['sum'] == 2)]['pid'], NSAMPLES, replace=False)
iph_sample = np.random.choice(df[(df['iph'] == 1) & (df['sum'] == 2)]['pid'], NSAMPLES, replace=False)
ivh_sample = np.random.choice(df[(df['ivh'] == 1) & (df['sum'] == 2)]['pid'], NSAMPLES, replace=False)
sah_sample = np.random.choice(df[(df['sah'] == 1) & (df['sum'] == 2)]['pid'], NSAMPLES, replace=False)
non_sample = np.random.choice(df[df['any'] != 1]['pid'], 5*NSAMPLES, replace=False)

def write_images(samples, savedir):
    if not os.path.exists(savedir): os.makedirs(savedir)
    for i in tqdm(samples, total=len(samples)):
        x = np.load('{}{}.npy'.format(NUMPYDIR, i))
        x = np.clip(x, 0, 80)
        x = x - np.min(x)
        x = x / np.max(x)
        x = x * 255
        x = x.astype('uint8')
        status = cv2.imwrite('{}{}.png'.format(savedir, i), x)

write_images(edh_sample, 'demo/edh/')
write_images(sdh_sample, 'demo/sdh/')
write_images(iph_sample, 'demo/iph/')
write_images(ivh_sample, 'demo/ivh/')
write_images(sah_sample, 'demo/sah/')
write_images(non_sample, 'demo/normal/')