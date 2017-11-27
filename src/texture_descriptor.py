#=============================================================
#
#
# Author: Saulo Pereira {scrps@cin.ufpe.br}
#=============================================================

from deap import gp
from opencv import cv2


#>Descriptor computation
img = cv2.imread('./../data/1.1.01.tiff', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Test', img)
cv2.waitKey(0)

width, height = cv2.GetSize(img)

# for i in 0
