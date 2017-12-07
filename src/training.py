#==================================================================================================
# Training/Evolution script
#
# Date created: 4/12/2017
# Code author: Saulo Pereira {scrps@cin.ufpe.br}
#==================================================================================================

import random

from deap import algorithms, gp, tools
import numpy as np

import gpcriptor as gpc

#>SCRIPT:------------------------------------------------------------------------------

#>Training parameters----------------------
kNClassesDataset =              112
kNTrainingClasses =             20      #TODO: 20
kClassesSize =                  100
kNTrainingInstances =           2       #instances per class
#>Algorithm parameters:
kCodeSize =                     7
kWindowSize =                   5
#>Evolution parameters:
kPopSize =                      50       #TODO: 100
kXOverRate =                    0.8
kMutRate =                      0.2
kElitRate =                     0.01
kMaxGenerations =               5
#------------------------------------------

#>Training setup
classes = range(1, kNClassesDataset+1)
del classes[14] #dataset problem

# Randomize classes to use on experiment
sample_classes = random.sample(classes, kNTrainingClasses)
sample_classes.sort()

# For each selected class, randomize the training instances
sample_instances = []
for i in sample_classes:
    sample_instances = sample_instances + \
        [(i, random.sample(range(0, kClassesSize/2), kNTrainingInstances))]
print sample_instances

#>Define Evolution framework
pset = gpc.CreatePrimitiveSet(kWindowSize, kCodeSize)
tbox = gpc.DefineEvolutionToolbox(pset, sample_instances, kCodeSize, kWindowSize)

pop = tbox.generate_population(kPopSize)
hof = tools.HallOfFame(1)

# Define Log structure
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(lambda ind: ind.height)
mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

# Evolutionary algorithm call
pop, log = algorithms.eaSimple(pop, tbox, kXOverRate, kMutRate, kMaxGenerations,
                               stats=mstats, halloffame=hof, verbose=True)

print '\nBest individual:\n', hof[0]


