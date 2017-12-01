#==================================================================================================
# Date created: 29/11/2017
# Code author: Saulo Pereira {scrps@cin.ufpe.br}
#==================================================================================================

# TODO: pesquisar ordem correta dos imports
from opencv import cv2
from deap import gp
import numpy as np

import gpcriptor as gpc
import image_processing as imp 

def FeatureExtraction(img, individual_lambda, code_size, window_size):
    kCS = code_size
    kWS = window_size

    img_WH = img.shape
    height_w = img_WH[0] - kWS/2
    width_w = img_WH[1] - kWS/2

    features = np.zeros(kCS**2, dtype=np.int)
    # iterate over image pixels to fill the feature vector (histogram) 
    for r in range(kWS/2, height_w):
        for c in range(kWS/2, width_w):
            window = imp.LinearWindow(img, kWS, (r,c))
            
            bs = np.array(individual_lambda(*window))
            bs = bs > 0.0
            print bs 
            
            bin = 0
            for bit in bs:
                bin = (bin << 1) | bit
            # print bin

            features[bin] = features[bin] + 1


    return features


#> TEST SCRIPT: ----------------------------------------------------------------------
#>Algorithm parameters:
# Constants
kCS = 2
kWS = 3

#>Input data:
img = cv2.imread('./../data/brodatz/D1.bmp', cv2.IMREAD_GRAYSCALE)
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
# cv2.imshow('Test Image', img)
# cv2.waitKey(0)


#>Pre processing:
#GP setting:
pset = gpc.CreatePrimitiveSet(kWS,kCS)
tbox = gpc.DefineEvolutionToolbox(pset)

rand_tree = tbox.generate_tree()
print 'Expression:\n', rand_tree
peval = gp.compile(rand_tree, pset)

#>Test cases:
print FeatureExtraction(img, peval, kCS, kWS)