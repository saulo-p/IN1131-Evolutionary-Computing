#=============================================================
# Image processing functions
# Date created: 27/11/2017
# Code author: Saulo Pereira {scrps@cin.ufpe.br}
#=============================================================

from opencv import cv2

#>Algorithm parameters
# Constants (TODO: use as arguments, not global variables)
kWS = 30


def LinearWindow(img, window_size, (row, col)):
    WS = window_size
    window = img[row-WS/2:row+WS/2, col-WS/2:col+WS/2]
    cv2.imshow('Window',window)
    cv2.waitKey(50)

# def ComputeFeatureVector()

#>Input data
img = cv2.imread('./../data/1.1.01.tiff', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Test', img)
cv2.waitKey(0)

WH = img.shape
height_w = WH[0] - kWS/2
width_w = WH[1] - kWS/2
# print width_w

# cv2.imshow('Crop', img[10:100, 10:100])
# cv2.waitKey(0)

for r in range(kWS/2, height_w):
    for c in range(kWS/2, width_w):
        LinearWindow(img, kWS, (r,c))

