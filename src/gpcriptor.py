#==================================================================================================
# Image Descriptor: A Genetic Programming Approach to Multiclass Texture Classification
# 
# Date created: 28/11/2017
# Code author: Saulo Pereira {scrps@cin.ufpe.br}
#==================================================================================================

import itertools
import operator

from deap import base, creator, gp, tools
import numpy as np
from opencv import cv2
from scipy.spatial.distance import pdist, squareform

import image_processing as imp 

#>About the training protocol:
#The dataset is divided between Training and Test sets.
#The Training set is randomly sampled to reduce the costs of evolution.
# The samples are the random (small) subsets of training used to classification and 
# fitness computation.

class InstancesFeatures:
    def __init__(self, n_instances, n_features):
        self.class_ids_ = np.zeros(shape=n_instances, dtype=int)
        self.class_instances_ = np.zeros(shape=n_instances, dtype=int)
        self.feature_matrix_ = np.zeros(shape=(n_features, n_instances), dtype=int)
        self.label_1nn_ = -1*np.ones(shape=n_instances, dtype=int)


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


    def featuresMatrix(self):
        return self.feature_matrix_


def protectedDiv(num, den):
    try: return num/den
    except ZeroDivisionError: return 0

def codeFunction(*args):
    return args

def CzekanowskiDistance(u, v):
    uv = np.matrix([u, v])
    uv = np.min(uv, axis=0)
    num = 2*np.sum(uv)
    
    den = np.sum(u) + np.sum(v)

    return 1.0 - 1.0*num/den

def FitnessEvaluation(rand_samples, n_tr_inst_p_class, code_size, window_size,
                        toolbox, individual):
    """Individual fitness evaluation. Based on the classification capabilities."""
    k_n_classes = len(rand_samples)
    #>Generate lambda expression of individual being evaluated
    ind_lambda = toolbox.compile(individual)

    #>Compute feature vectors of the whole test (training + eval) set based on
    #  current individual and store the results on "test_set".
    base_path = 'C:/Users/Saulo/Documents/GitHub/IN1131-Evolutionary-Computing/data/brodatz/resampled/D'
    # feature_vectors = []
    test_set = InstancesFeatures(k_n_classes*n_tr_inst_p_class, 2**code_size)
    test_idx = 0
    for r_s in rand_samples:
        #TODO: Talvez trocar o range por percorrer as instancias do rand_samples
        for i in range(0, n_tr_inst_p_class):
            if (i < 10):
                str_idx = '0' + str(i)
            else:
                str_idx = str(i)

            # Read patch image
            img_path = base_path + str(r_s[0]) + '_' + str_idx[0]  + '_' + str_idx[1] + '.bmp'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Compute patch feature vector 
            fv = FeatureExtraction(img, ind_lambda, code_size, window_size)
        
            # Fill the feature vector matrix
            test_set.addInstance(test_idx, r_s[0], i, fv)
            test_idx = test_idx + 1
            # feature_vectors = feature_vectors + [((r_s[0], i), fv)]
    # print feature_vectors

    #>Compute pdist and label each individual using 1NN
    D = squareform(pdist(test_set.featuresMatrix().transpose()))
    D_cze = squareform(pdist(test_set.featuresMatrix().transpose(), CzekanowskiDistance))
    # print D

    #>Classify sampled instances using 1NN and computes cluster distances
    db = 0.0
    dw = 0.0
    for i in range(0, k_n_classes):
        for inst in rand_samples[i][1]:
            # Resolve indexing
            idx_d = i*n_tr_inst_p_class + inst
            
            #>Accuracy
            # Euclidean distances between current instance and whole set
            dists_z = D[idx_d]
            dists = np.delete(dists_z, i)
            min_idx = np.argmin(dists)
            # Correct the index shift due to removal of self distance
            if (min_idx >= idx_d):
                min_idx = min_idx + 1
            # Compute and store individual label
            label = min_idx / n_tr_inst_p_class
            test_set.labelInstance(idx_d, rand_samples[label][0])

            #>Distance
            dists_cze_z = D_cze[idx_d] 
            # Separates the distances
            d_same = dists_cze_z[i*n_tr_inst_p_class:(i+1)*n_tr_inst_p_class]
            d_diff = np.delete(dists_cze_z,
                            range(i*n_tr_inst_p_class,
                                  (i+1)*n_tr_inst_p_class))

            db = db + np.min(d_diff)/(k_n_classes*n_tr_inst_p_class)
            dw = dw + np.max(d_same)/(k_n_classes*n_tr_inst_p_class)


    print 'End of Classification:'
    print test_set.correctClassifications()

    accuracy = test_set.correctClassifications()[1]
    distance = 1.0/(1 + np.exp(-5.0*(db - dw)))

    return (1.0 - (accuracy + distance)/2.0, )

def CreatePrimitiveSet (window_size, code_size):
    """TODO: Talvez essa funcao deva ficar como script com import organizado..."""
    #About primitives:
    # input types requires length (use lists)
    # input types use list but arguments are seen as separate elements
    # return type must be a class.
    # return type must be hashable, so lists which are dynamic elements are not allowed.
    
    kCS = code_size
    kWS = window_size

    #>Function set (terminals come via evaluation)
    pset = gp.PrimitiveSetTyped("GP-criptor", itertools.repeat(float, kWS**2), tuple, "Px")
    # Arithmetic operations
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)
    # Root node
    pset.addPrimitive(codeFunction, [float]*kCS, tuple)

    return pset

def DefineEvolutionToolbox (primitive_set, training_instances, eval_instances,
    code_size, window_size):
    """TODO: Parameterize this function so it receives the evolution parameters 
    from file/struct"""
    #>Evolution parameters:
    kTreeMinDepth = 2
    kTreeMaxDepth = 5           #TODO: 10 
    #TODO: create enum for categorical parameters
    # kInitialization = 'genHalfAndHalf'
    # kSelection = 'selTournament'
    kTournamentSize = 2         #TODO: 7

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

    tbox = base.Toolbox()
    tbox.register("generate_expr", gp.genHalfAndHalf, pset=primitive_set, 
                  min_=kTreeMinDepth, max_=kTreeMaxDepth)
    tbox.register("generate_ind_tree", tools.initIterate, creator.Individual, 
                  tbox.generate_expr)
    tbox.register("generate_population", tools.initRepeat, list, tbox.generate_ind_tree)
    tbox.register("compile", gp.compile, pset = primitive_set)
    tbox.register("evaluate", FitnessEvaluation, training_instances, eval_instances,
        code_size, window_size, tbox )
    tbox.register("select", tools.selTournament, tournsize=kTournamentSize)
    tbox.register("mate", gp.cxOnePoint)
    tbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    tbox.register("mutate", gp.mutUniform, expr=tbox.expr_mut, pset=primitive_set)
    #enforce size constraint over generated individuals
    tbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                         max_value=kTreeMaxDepth))
    tbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), 
                                           max_value=kTreeMaxDepth))

    return tbox    

def FeatureExtraction(img, individual_lambda, code_size, window_size):
    """TODO: Create struct algorithm parameters to encapsulate CS and WS"""
    kNF = 2**code_size
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


#> TEST SCRIPT: ----------------------------------------------------------------------
# from opencv import cv2

# #>Algorithm parameters:
# # Constants
# kCS = 2
# kWS = 3

# #>Input data:
# img = cv2.imread('./../data/brodatz/resampled/D1_0_0.bmp', cv2.IMREAD_GRAYSCALE)
# img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
# # cv2.imshow('Test Image', img)
# # cv2.waitKey(0)


# #>Pre processing:
# #GP setting:
# pset = CreatePrimitiveSet(kWS,kCS)
# tbox = DefineEvolutionToolbox(pset)

# rand_tree = tbox.generate_tree()
# print 'Expression:\n', rand_tree
# peval = gp.compile(rand_tree, pset)

# #>Test cases:
# print FeatureExtraction(img, peval, kCS, kWS)


#>TESTS: -----------------------------------------------------
# #>Evolution toolbox
# pset = createPrimitiveSet(3,4)
# tbox = base.Toolbox()
# tbox.register("generate_expr", gp.genFull, pset, min_=2, max_=3)
# tbox.register("generate_tree", tools.initIterate, gp.PrimitiveTree, tbox.generate_expr)

# # Print individual generated from toolbox interface
# expr = tbox.generate_tree()
# print expr
# peval = gp.compile(expr,pset)
# print 'List: ', peval(0,1,2,3,4,5,6,7,8)
