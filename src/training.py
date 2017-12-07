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
#>Statistics parameters----------------------
kNRoundsClasses =               10
kNRoundsInstances =             10
#>Training parameters
kNClassesDataset =              112
kNTrainingClasses =             15
kClassesSize =                  100
kNTrainingInstances =           2       #instances per class
#>Algorithm parameters:
kCodeSize =                     5
kWindowSize =                   5
#>Evolution parameters:
kPopSize =                      25
kXOverRate =                    0.8
kMutRate =                      0.2
kElitRate =                     0.01
kMaxGenerations =               5
#------------------------------------------

#>Training setup
classes = range(1, kNClassesDataset+1)
del classes[14] #dataset problem

#>ITERATE OVER CLASSES
for n_cl in range(0, kNRoundsClasses):

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
    # print '\nGenerating Primitive Set...\n'
    pset = gpc.CreatePrimitiveSet(kWindowSize, kCodeSize)
    # print '\nDefining the Toolbox...\n'
    tbox = gpc.DefineEvolutionToolbox(pset, sample_instances, kCodeSize, kWindowSize)

    #>Define Log structure
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    #>ITERATE OVER INSTANCES
    for n_in in range(0, kNRoundsInstances):

        # print '\nGenerating Population...\n'
        pop = tbox.generate_population(kPopSize)
        hof = tools.HallOfFame(1)

        # Evolutionary algorithm call
        # print '\nExecuting the evolutionary process...\n'
        pop, log = algorithms.eaSimple(pop, tbox, kXOverRate, kMutRate, kMaxGenerations,
                                    stats=mstats, halloffame=hof, verbose=True)

        print '\nBest individual:\n', hof[0]

        fh = open('./results/' + str(n_cl) + '_' + str(n_in) '.txt', 'w')
        fh.write(log)
        fh.write('\n')
        fh.write('\nBest individual:\n')
        fh.write(hof[0])
        fh.close()
