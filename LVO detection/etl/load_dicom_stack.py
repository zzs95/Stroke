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
    array *= float(loaded[0].RescaleSlope)
    array += float(loaded[0].RescaleIntercept)
    return array, loaded[0].ImageType
 -

ZIPFILE_DIR = '/media/user1/Seagate Desktop Drive/DATA/zips/'
WORKING_DIR = '/home/user1/cta/'

zipfiles = glob.glob(os.path.join(ZIPFILE_DIR, '*.zip'))

for z in zipfiles:
    # 
    os.system('unzip {} {}'.format(z, savedir))

