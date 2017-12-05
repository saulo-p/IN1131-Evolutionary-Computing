#==================================================================================================
# Training/Evolution script
#
# Date created: 4/12/2017
# Code author: Saulo Pereira {scrps@cin.ufpe.br}
#==================================================================================================

import random

from deap import algorithms, tools
import numpy as np

import gpcriptor as gpc

#>SCRIPT:------------------------------------------------------------------------------

#>Training parameters----------------------
kNClasses = 112
kNTrainingClasses =     3       #TODO: 20
kClassesSize =          100
kNEvalInstances =       10      
kNTrainingInstances =   2
#>Algorithm parameters:
kCodeSize =             2       #TODO: 7
kWindowSize =           3       #TODO: 5
#>Evolution parameters:
kPopSize =              3       #TODO: 100
kXOverRate =            0.8
kMutRate =              0.2
kElitRate =             0.01
kMaxGenerations =       3       #TODO: 30
#------------------------------------------

#>Training setup
classes = range(1, kNClasses+1)
del classes[14] #dataset problem

# Randomize classes to use on experiment
sample_classes = random.sample(classes, kNTrainingClasses)
sample_classes.sort()

# For each selected class, randomize the training instances
classes_samples = []
for i in sample_classes:
    classes_samples = classes_samples + \
        [(i, random.sample(range(0, kNEvalInstances),\
         kNTrainingInstances))]
print classes_samples


#>Define Evolution framework
pset = gpc.CreatePrimitiveSet(kWindowSize, kCodeSize)
tbox = gpc.DefineEvolutionToolbox(pset, classes_samples, kNEvalInstances, kCodeSize, kWindowSize)

pop = tbox.generate_population(kPopSize)
best = tools.HallOfFame(1)

# Define Log structure
# stats_classrate = tools.Statistics(lambda ind: ind.fitness.values)

# Evolutionary algorithm call
pop, log = algorithms.eaSimple(pop, tbox, kXOverRate, kMutRate, kMaxGenerations,
                               #stats=mstats, 
                               halloffame=best, verbose=True)

