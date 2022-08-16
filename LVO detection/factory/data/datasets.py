import pandas as pd
import numpy as np
import pydicom
import torch
import glob, os.path as osp
import cv2

from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, Sampler


class SeriesDataset(Dataset):


    def __init__(self, 
                 imgfiles, 
                 labels, 
                 pad, 
                 resize, 
                 crop,
                 transform, 
                 preprocessor,
                 hu_augment=0,
                 num_slices=64,
                 volume_ratio=1,
                 resample=False,
                 crop_center=False,
                 flip=False,
                 reverse=False,
                 test_mode=False):

        self.imgfiles = imgfiles
        self.labels = labels
        self.pad = pad
        self.resize = resize
        self.crop = crop
        self.transform = transform
        self.preprocessor = preprocessor
        self.hu_augment = hu_augment
        self.num_slices = num_slices
        self.volume_ratio = volume_ratio
        self.resample = resample
        self.crop_center = crop_center
        self.flip = flip
        self.reverse = reverse
        self.test_mode = test_mode

        assert volume_ratio <= 1 and volume_ratio > 0, '`volume_ratio` must be in range (0, 1]'

        self.badfiles = []

    def __len__(self):
        return len(self.imgfiles)


    def fit_to_num_slices(self, X):
        if X.shape[0] > self.num_slices:
            X = X[(X.shape[0]-self.num_slices):]
        if X.shape[0] < self.num_slices:
            empty_slices = np.zeros_like(X[0])
            empty_slices = np.repeat(np.expand_dims(empty_slices, axis=0), self.num_slices-X.shape[0], axis=0)
            empty_slices[...] = np.min(X)
            X = np.vstack((X, empty_slices))
        assert X.shape[0] == self.num_slices
        return X


    def cut_volume(self, X):
        if self.resample:
            X = zoom(X, [self.volume_ratio, 1, 1], order=0, prefilter=False)
        else:
            start = int(X.shape[0] * (1.-self.volume_ratio))
            X = X[start:]
        return X


    @staticmethod
    def _crop_center(img):
        h, w = img.shape[1:]
        if h == w:
            return img
        croph = cropw = min(h,w)
        starth = h//2-(croph//2)
        startw = w//2-(cropw//2)    
        return img[:,starth:starth+croph,startw:startw+cropw]


    def process_image(self, X):
        # Images will be of shape (Z, H, W)
        if self.crop_center: X = self._crop_center(X)
        if self.volume_ratio < 1: X = self.cut_volume(X)
        X = np.expand_dims(X, axis=-1)
        if self.hu_augment > 0 and not self.test_mode:
            X = X + int(np.random.normal(0, self.hu_augment))
        # shape = (Z, H, W, 1)
        if self.pad: X = np.asarray([self.pad(_) for _ in X]) 
        if self.resize: X = np.asarray([self.resize(image=_)['image'] for _ in X]) 
        if self.crop: 
            to_crop = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_crop.update({'image': X[0]})
            cropped = self.crop(**to_crop)
            X = np.asarray([cropped['image']] + [cropped['image{}'.format(_)] for _ in range(1,len(X))])
        if self.transform: 
            to_transform = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_transform.update({'image': X[0]})
            transformed = self.transform(**to_transform)
            X = np.asarray([transformed['image']] + [transformed['image{}'.format(_)] for _ in range(1,len(X))])
        if self.preprocessor: X = self.preprocessor.preprocess(X)

        X = self.fit_to_num_slices(X)

        return X.transpose(3,0,1,2)


    def get(self, i):
        try:
            return np.load(self.imgfiles[i])
        except Exception as e:
            print(e)
            return None


    def __getitem__(self, i):
        while i in self.badfiles:
            i = np.random.choice(range(len(self.imgfiles)))
        X = self.get(i)
        while type(X) == type(None):
            print('Failed to read {}'.format(self.imgfiles[i]))
            self.badfiles.append(i)
            i = np.random.choice(range(len(self.imgfiles)))
            X = self.get(i)
        X = self.process_image(X)

        if not self.test_mode:
            if self.flip:
                # X.shape = (C, Z, H, W)
                mode = np.random.randint(4)
                if mode == 1: # horizontal flip
                    X = X[:,:,:,::-1]
                elif mode == 2: # vertical flip
                    X = X[:,:,::-1,:]
                elif mode == 3: # both flips
                    X = X[:,:,::-1,::-1]
                X = np.ascontiguousarray(X)
            if self.reverse:
                if np.random.binomial(1, 0.5):
                    X = X[:,::-1]
                    X = np.ascontiguousarray(X)

        # Torchify 
        X = torch.tensor(X).float()
        y = torch.tensor(self.labels[i]).long()
        return X, y



class DICOMDataset(Dataset):


    def __init__(self, 
                 imgfiles, 
                 labels, 
                 pad, 
                 resize, 
                 crop,
                 transform, 
                 preprocessor,
                 num_slices=64,
                 test_mode=False):

        # imgfiles is actually a list of folders containing DICOM series
        self.imgfiles = imgfiles
        self.labels = labels
        self.pad = pad
        self.resize = resize
        self.crop = crop
        self.transform = transform
        self.preprocessor = preprocessor
        self.num_slices = num_slices
        self.test_mode = test_mode

        self.badfiles = []

    def __len__(self):
        return len(self.imgfiles)


    def fit_to_num_slices(self, X):
        if X.shape[0] < self.num_slices:
            empty_slices = np.zeros_like(X[0])
            empty_slices = np.repeat(np.expand_dims(empty_slices, axis=0), self.num_slices-X.shape[0], axis=0)
            empty_slices[...] = np.min(X)
            X = np.vstack((X, empty_slices))
        assert X.shape[0] == self.num_slices
        return X


    def resample_z(self, X):
        return zoom(X, [self.num_slices/X.shape[0], 1, 1], order=0, prefilter=False)


    def process_image(self, X):
        # Images will be of shape (Z, H, W)
        if X.shape[0] > self.num_slices:
            X = self.resample_z(X)
        X = np.expand_dims(X, axis=-1)
        # shape = (Z, H, W, 1)
        if self.pad: X = np.asarray([self.pad(_) for _ in X]) 
        if self.resize: X = np.asarray([self.resize(image=_)['image'] for _ in X]) 
        if self.crop: 
            to_crop = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_crop.update({'image': X[0]})
            cropped = self.crop(**to_crop)
            X = np.asarray([cropped['image']] + [cropped['image{}'.format(_)] for _ in range(1,len(X))])
        if self.transform: 
            to_transform = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_transform.update({'image': X[0]})
            transformed = self.transform(**to_transform)
            X = np.asarray([transformed['image']] + [transformed['image{}'.format(_)] for _ in range(1,len(X))])
        if self.preprocessor: X = self.preprocessor.preprocess(X)

        X = self.fit_to_num_slices(X)

        return X.transpose(3,0,1,2)


    @staticmethod
    def dcmread(f):
        try:
            attributes = ['ImageType', 'ImagePositionPatient', 'ImageOrientationPatient',
                          'InstanceNumber', 'RescaleSlope', 'RescaleIntercept', 'pixel_array']
            dcm = pydicom.dcmread(f)
            for at in attributes:
                assert hasattr(dcm, at)
            return dcm
        except Exception as e:
            #print(e)
            return None


    @staticmethod
    def determine_plane(o):
        o = [round(x) for x in o]
        plane = np.cross(o[0:3], o[3:6])
        plane = [abs(x) for x in plane]
        return np.argmax(plane)


    def load_dicom_stack(self, i, mismatch='position'):
        files = glob.glob(osp.join(self.imgfiles[i], '*'))
        loaded = np.asarray([self.dcmread(f) for f in files])
        loaded = np.asarray([l for l in loaded if type(l) != type(None)])
        if len(loaded) == 0: return None
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
        # Determine plane using ImageOrientationPatient
        pos = self.determine_plane(loaded[0].ImageOrientationPatient)
        positions = np.asarray([l.ImagePositionPatient for l in loaded]).astype('float32')
        positions = positions[:,pos]
        instances = [float(l.InstanceNumber) for l in loaded]
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


    def get(self, i):
        try:
            return self.load_dicom_stack(i)
        except Exception as e:
            print(e)
            return None


    def __getitem__(self, i):
        # while i in self.badfiles:
        #     i = np.random.choice(range(len(self.imgfiles)))
        X = self.get(i)
        # while type(X) == type(None):
        #     #print('Failed to read {}'.format(self.imgfiles[i]))
        #     self.badfiles.append(i)
        #     i = np.random.choice(range(len(self.imgfiles)))
        #     X = self.get(i)
        if type(X) == type(None):
            return torch.tensor(0).float(), torch.tensor(0).float()

        X = self.process_image(X)

        # Torchify 
        X = torch.tensor(X).float()
        y = torch.tensor(self.labels[i]).long()
        return X, y


class BalancedELVOSampler(Sampler):


    def __init__(self,
        dataset,
        labels,
        weights=None):

        super().__init__(data_source=dataset)
        if len(np.unique(labels)) == 2:
            if type(weights) == type(None):
                weights = [2./3, 1./3]
            weights /= np.sum(weights)
            self.num_samples_per_stratum = {
                0: int(weights[0]*len(labels)),
                1: int(weights[1]*len(labels))
            }
        elif len(np.unique(labels)) == 3:
            if type(weights) == type(None):
                weights = [0.5, 0.35, 0.15]
            weights /= np.sum(weights)            
            self.num_samples_per_stratum = {
                0: int(0.50*len(labels)),
                1: int(0.35*len(labels)),
                2: int(0.15*len(labels)),
            }
        self.indices = {
            k : np.where(labels == k)[0] for k in self.num_samples_per_stratum.keys()
        }
        self.length = np.sum(list(self.num_samples_per_stratum.values()))


    def __iter__(self):
        indices = []
        for k,v in self.num_samples_per_stratum.items():
            indices.extend(list(np.random.choice(self.indices[k], v, replace=v>len(self.indices[k]))))
        shuffled = np.random.permutation(indices)
        return iter(shuffled.tolist())


    def __len__(self):
        return self.length


# class BalancedSampler(Sampler):


#     def __init__(self,
#         dataset,
#         strata):
#     #
#     # strata : dict 
#     #    - key : stratum
#     #    - value : indices belonging to stratum
#     #
#         super(BalancedSampler, self).__init__(data_source=dataset)
#         self.strata = strata
#         length = np.sum([len(v) for k,v in strata.items()])
#         self.num_samples_per_stratum = int(length / len(strata.keys()))
#         self.length = self.num_samples_per_stratum * len(strata.keys())


#     def __iter__(self):
#         # Equal number per stratum
#         # Custom number per stratum will require additional code
#         indices = [] 
#         for k,v in self.strata.items():
#             indices.append(np.random.choice(v, self.num_samples_per_stratum, replace=len(v) < self.num_samples_per_stratum))
#         shuffled = np.random.permutation(np.concatenate(indices))
#         return iter(shuffled.tolist())


#     def __len__(self):
#         return self.length




