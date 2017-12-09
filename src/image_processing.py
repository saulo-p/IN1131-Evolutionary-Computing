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

def TESTofScope():
    return 0

__all__ = ['LinearWindow']
