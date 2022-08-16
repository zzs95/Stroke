import numpy as np
import matplotlib.pyplot as plt
import glob


def view(x, nrows=4, ncols=4):
    indices = np.linspace(0, len(x)-1, 16) 
    for ind, i in enumerate(indices):
        plt.subplot(4,4,ind+1)
        plt.imshow(x[int(i)], cmap='gray')
    plt.show()


def window(x, WW=100, WL=50):
    x = np.clip(x, WL-WW/2, WL+WW/2)
    x -= np.min(x)
    x /= np.max(x)
    return (x*255).astype('uint8')


numpys = glob.glob('/users/ipan/downloads/mips/*/*.npy')
numpys = [np.load(_) for _ in numpys]

for i in range(len(numpys)):
    arr = window(numpys[i])
    view(arr)



