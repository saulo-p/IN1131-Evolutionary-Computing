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
from scipy.spatial.distance import pdist

import image_processing as imp 

#>Functions (TODO: create separate file if necessary)
def protectedDiv(num, den):
    try: return num/den
    except ZeroDivisionError: return 0

def codeFunction(*args):
    return args
#---------------------------------------------------
#About primitives:
# input types requires length (use lists)
# input types use list but arguments are seen as separate elements
# return type must be a class.
# return type must be hashable, so lists which are dynamic elements are not allowed.

def FitnessEvaluation(training_instances, eval_instances, code_size, window_size, toolbox, individual):
    """Individual fitness evaluation. Based on the classification capabilities."""
    #>Generate lambda expression of individual being evaluated
    ind_lambda = toolbox.compile(individual)

    #>Compute feature vectors of the whole test set based on current individual
    base_path = 'C:\Users\Saulo\Documents\GitHub\IN1131-Evolutionary-Computing\data\\brodatz\\resampled\D'
    feature_vectors = []
    test_set = []
    eval_set = []
    for t_i in training_instances:
        for i in range(0, eval_instances):
            if (i < 10):
                str_idx = '0' + str(i)
            else:
                str_idx = str(i)

            # Read patch image
            img_path = base_path + str(t_i[0]) + '_' + str_idx[0]  + '_' + str_idx[1] + '.bmp'
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            fv = FeatureExtraction(img, ind_lambda, code_size, window_size)
        
            # Separate test and evaluation set
            if (i in t_i[1]):
                test_set = test_set + [((t_i[0], i), fv)]
            else:
                eval_set = eval_set + [((t_i[0], i), fv)]

            feature_vectors = feature_vectors + [((t_i[0], i), fv)]

    print feature_vectors         

    #>Compute pdist between each individual of test set and whole evaluation set
    for test_i in test_set:
        #Create set with current individual + eval_set
        #pdist()
        print test_i
    print 'bb'
    #Label the individual



def CreatePrimitiveSet (window_size, code_size):
    """TODO: Talvez essa funcao deva ficar como script com import organizado..."""
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
