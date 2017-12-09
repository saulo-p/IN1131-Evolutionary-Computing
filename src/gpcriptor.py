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

import features_class as fc


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


def FitnessEvaluation(train_samples, code_size, window_size, toolbox, individual):
    """Individual fitness evaluation. Based on the classification capabilities."""
    kNClasses = len(train_samples)
    kNInstances = len(train_samples[1])

    #>Generate lambda expression of individual being evaluated
    ind_lambda = toolbox.compile(individual)

    #>Compute feature vectors for all the images in the training set based on
    #  current individual lambda and store the results on "train_set".
    train_set = fc.InstancesFeatures(kNClasses*kNInstances, 2**code_size)
    train_set.populate(ind_lambda, train_samples, window_size)

    #>Compute pdist and label each individual using 1NN
    D = squareform(pdist(train_set.featuresMatrix().transpose()))
    D_cze = squareform(pdist(train_set.featuresMatrix().transpose(), CzekanowskiDistance))

    #>Classify sampled instances using 1NN and computes cluster distances
    db = 0.0
    dw = 0.0
    for i in range(0, len(D)):
        #>Accuracy (fitness term)
        dists_z = D[i]
        dists = np.delete(dists_z, i)
        min_idx = np.argmin(dists)
        # Correct the index shift due to removal of self distance
        if (min_idx >= i):
            min_idx = min_idx + 1
        # Compute and store individual label
        label_idx = min_idx / kNInstances
        train_set.labelInstance(i, train_samples[label_idx][0])

        #>Distance (fitness term)
        dists_cze_z = D_cze[i]
        mod_i = i/kNInstances
        # Separates the distances in blocks from same class and others
        d_same = dists_cze_z[mod_i*kNInstances:(mod_i+1)*kNInstances]
        d_diff = np.delete(dists_cze_z, range(mod_i*kNInstances, (mod_i+1)*kNInstances))
        # Update the distance counters       
        db = db + np.min(d_diff)
        dw = dw + np.max(d_same)

    db = db/(kNClasses*kNInstances)
    dw = dw/(kNClasses*kNInstances)

    # print 'End of Classification:'
    # print train_set.correctClassifications()

    accuracy = train_set.correctClassifications()[1]
    distance = 1.0/(1 + np.exp(-5.0*(db - dw)))

    fitness = 1.0 - (accuracy + distance)/2.0

    return (fitness, )


def CreatePrimitiveSet (window_size, code_size):
    """SHIT!"""
    #About primitives:
    # input types requires length (use lists)
    # input types use list but arguments are seen as separate elements
    # return type must be a class.
    # return type must be hashable, so lists which are dynamic elements are not allowed.
    kCS = code_size
    kWS = window_size

    pset = gp.PrimitiveSetTyped("GP-criptor", itertools.repeat(float, kWS**2), tuple, "P")
    pset.addPrimitive(codeFunction, [float]*kCS, tuple)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)

    return pset


def DefineEvolutionToolbox (primitive_set, training_instances, code_size, window_size):
    """TODO: Parameterize this function so it receives the evolution parameters 
    from file/struct"""
    #>Evolution parameters:
    #TODO: create enum for categorical parameters
    # kInitialization = 'genHalfAndHalf'
    # kSelection = 'selTournament'
    kTreeMinDepth = 2
    kTreeMaxDepth = 7
    kTournamentSize = 7

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

    tbox = base.Toolbox()
    tbox.register("generate_expr", gp.genHalfAndHalf, pset=primitive_set, 
                  min_=kTreeMinDepth, max_=kTreeMaxDepth)
    tbox.register("generate_ind_tree", tools.initIterate, creator.Individual, 
                  tbox.generate_expr)
    tbox.register("generate_population", tools.initRepeat, list, tbox.generate_ind_tree)
    tbox.register("compile", gp.compile, pset=primitive_set)
    tbox.register("evaluate", FitnessEvaluation, training_instances, code_size,
                    window_size, tbox )
    tbox.register("select", tools.selTournament, tournsize=kTournamentSize)
    tbox.register("mate", gp.cxOnePoint)
    tbox.register("expr_mut", gp.genFull, min_=1, max_=3, type_=float)
    tbox.register("mutate", gp.mutUniform, expr=tbox.expr_mut, pset=primitive_set)
    #enforce size constraint over generated individuals
    tbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                         max_value=kTreeMaxDepth))
    tbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), 
                                           max_value=kTreeMaxDepth))

    return tbox    

