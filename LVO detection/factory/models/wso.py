import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WSO(nn.Module):
    

    def __init__(self, 
                 wl, 
                 ww, 
                 act='sigmoid', 
                 upper=255.0, 
                 dim=3):

        super(WSO, self).__init__()

        if type(wl) != list: wl = [wl]
        if type(ww) != list: ww = [ww]

        assert len(wl) == len(ww)
        self.nch = len(wl)

        self.wl  = wl 
        self.ww  = ww
        self.act = act 
        self.upper = upper
        
        conv_module = nn.Conv3d if dim == 3 else nn.Conv2d
        self.conv = conv_module(1, self.nch, kernel_size=1, stride=1)
        self.init_weights()
    

    def forward(self, x):
        x = self.conv(x)
        if self.act == 'relu':
            x = F.relu(x)
            x[x > self.upper] = self.upper
        elif self.act == 'sigmoid': 
            x = torch.sigmoid(x) * self.upper
        return x
    

    def init_weights(self):
        params = self.get_params()
        self.conv.state_dict()['weight'].copy_(params[0].reshape(self.conv.weight.shape))
        self.conv.state_dict()['bias'].copy_(params[1].reshape(self.conv.bias.shape))
    

    def get_params(self, smooth=1.):
        ws = []
        bs = []
        for i in range(self.nch):
            if self.act == 'relu':
                w = self.upper / self.ww[i]
                b = -1. * self.upper * (self.wl[i] - self.ww[i] / 2.) / self.ww[i]
            elif self.act == 'sigmoid':
                w = 2./self.ww[i] * np.log(self.upper/smooth - 1.)
                b = -2.*self.wl[i]/self.ww[i] * np.log(self.upper/smooth - 1.)
            ws.append(w)
            bs.append(b)
        return torch.tensor(ws, requires_grad=True), \
               torch.tensor(bs, requires_grad=True)


## // TEST WSO
if __name__ == '__main__':
    import matplotlib as mpl ; mpl.use('tkAgg')
    import matplotlib.pyplot as plt
    import numpy as np 

    import pydicom

    def get_image_from_dicom(dicom_file, hu_min=0, hu_max=100):
        dcm = pydicom.read_file(dicom_file)
        array = dcm.pixel_array
        try:
            array = array * int(dcm.RescaleSlope)
            array = array + int(dcm.RescaleIntercept)
        except:
            pass
        array = array.astype("float32")
        array = np.clip(array, hu_min, hu_max)
        array -= np.min(array)
        array /= np.max(array)
        array *= 255.
        return array.astype('uint8')

    dicom_file = 'CT000010.dcm'
    test = pydicom.read_file(dicom_file)
    img = test.pixel_array
    img = img + int(test.RescaleIntercept)
    img = img * int(test.RescaleSlope)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    img = img.expand(-1,-1,3,-1,-1)
    wso = WSO(nch=2, wl=[40,50], ww=[80,130], act='relu')

    test1 = wso(img.float())
    test2 = get_image_from_dicom(dicom_file, 0, 100)
    plt.subplot(1,2,1)
    plt.imshow(test1[0,0,2].detach().numpy(), cmap='gray'); 
    plt.subplot(1,2,2)
    plt.imshow(test2, cmap='gray')
    plt.show()










