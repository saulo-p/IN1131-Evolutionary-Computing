from opencv import cv2
import numpy as np

import gpcriptor as gpc
import image_processing as imp

class InstancesFeatures:
    def __init__(self, n_instances, n_features):
        self.class_ids_ = np.zeros(shape=n_instances, dtype=int)
        self.class_instances_ = np.zeros(shape=n_instances, dtype=int)
        self.feature_matrix_ = np.zeros(shape=(n_features, n_instances), dtype=int)
        self.label_1nn_ = -1*np.ones(shape=n_instances, dtype=int)

    def populate(self, ind_lambda, sample_instances, window_size):
        base_path = 'C:/Users/Saulo/Documents/GitHub/IN1131-Evolutionary-Computing/data/brodatz/resampled/D'
        set_idx = 0
        for s_i in sample_instances:

            for inst in s_i[1]:
                str_idx = indexToString(inst)

                # Read patch image
                img_path = base_path + str(s_i[0]) + '_' + str_idx[0]  + '_' + str_idx[1] + '.bmp'
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                #TODO: talvez normalizar imagem de entrada.

                # Compute patch feature vector
                shape = self.feature_matrix_.shape
                fv = FeatureExtraction(img, ind_lambda, shape[0], window_size)

                # Fill the feature vector matrix
                self.addInstance(set_idx, s_i[0], inst, fv)
                set_idx = set_idx + 1

    def addInstance(self, idx, class_id, class_instance, features):
        self.class_ids_[idx] = class_id
        self.class_instances_[idx] = class_instance
        self.feature_matrix_[:,idx] = features

    def labelInstance(self, idx, label):
        self.label_1nn_[idx] = label

    def correctClassifications(self):
        labels = np.zeros(shape=len(self.label_1nn_), dtype=int)
        correct = 0
        total = 0
        for i in range(0, len(labels)):
            if (self.label_1nn_[i] == -1):
                labels[i] = -1
            else:
                total = total + 1
                if self.class_ids_[i] == self.label_1nn_[i]:
                    labels[i] = 1
                    correct = correct + 1

        acc = (1.0*correct)/total

        return (labels, acc)

    def numInstances(self):
        return len(self.class_ids_)

    def featuresVector(self, col_idx):
        return self.feature_matrix_[:, col_idx]

    def featuresMatrix(self):
        return self.feature_matrix_


def indexToString(inst_idx):
    if (inst_idx < 10):
        str_idx = '0' + str(inst_idx)
    else:
        str_idx = str(inst_idx)
    
    return str_idx

def FeatureExtraction(img, individual_lambda, features_len, window_size):
    """TODO: Add comment"""
    kNF = features_len
    kWS = window_size

    img_WH = img.shape
    height_w = img_WH[0] - kWS/2
    width_w = img_WH[1] - kWS/2

    features = np.zeros(kNF, dtype=np.int)
    # iterate over image pixels to fill the feature vector (histogram) 
    for r in range(kWS/2, height_w):
        for c in range(kWS/2, width_w):
            window = imp.LinearWindow(img, kWS, (r,c))
            
            bs = np.array(individual_lambda(*window))
            bs = bs > 0.0
            # print bs 
            
            bin = 0
            for bit in bs:
                bin = (bin << 1) | bit
            # print bin

            features[bin] = features[bin] + 1

    return features