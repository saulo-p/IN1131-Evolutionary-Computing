#=============================================================
# Image processing functions
# Date created: 27/11/2017
# Code author: Saulo Pereira {scrps@cin.ufpe.br}
#=============================================================

import numpy as np
from opencv import cv2

def LinearWindow(img, window_size, (row, col)):
    WS = window_size
    window = img[row-WS/2:row+WS/2+1, col-WS/2:col+WS/2+1]
    # cv2.imshow('Window',window)
    # cv2.waitKey(50)
    return window.reshape(1, WS**2)[0]


__all__ = ['LinearWindow']

#> Script portion: ----------------------------------------------

#>Data preparation 
# Resample Brodatz input images
data_path = './../data/brodatz/'

WS = 64
for i in range(1,112+1):
    if (i == 14):
        continue

    img = cv2.imread(data_path + 'D' + str(i) +'.bmp', cv2.IMREAD_GRAYSCALE)

    for rb in range(0, 10):
        for cb in range(0, 10):
            img_block = img[rb*WS:(rb+1)*WS, cb*WS:(cb+1)*WS]
            # cv2.imshow('Test Image', img_block)
            # cv2.waitKey(0)

            cv2.imwrite(data_path + 'resampled/D' +
              str(i) + '_' + str(rb) + '_' + str(cb) + '.bmp' , img_block)