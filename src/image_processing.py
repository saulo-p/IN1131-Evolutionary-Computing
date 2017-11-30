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

#> TESTS: -----------------------------------------------------

# kWS = 5

# #>Input data
# img = cv2.imread('./../data/1.1.01.tiff', cv2.IMREAD_GRAYSCALE)
# # cv2.imshow('Test', img)
# # cv2.waitKey(0)

# WH = img.shape
# height_w = WH[0] - kWS/2
# width_w = WH[1] - kWS/2
# # print width_w
