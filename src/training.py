#==================================================================================================
# Training/Evolution and Testing script
#
# Date created: 4/12/2017
# Code author: Saulo Pereira {scrps@cin.ufpe.br}
#==================================================================================================

'''>About the training protocol:
*The dataset is divided between Training and Test sets.
*The Training set is randomly sampled to reduce the costs of evolution.
*The samples are the random (small) subsets of training used to classification and 
fitness computation.'''

#import pickle
import random

from deap import algorithms, gp, tools
import numpy as np

import gpcriptor as gpc
import features_class as fc


def ComputeAccuracyOverTestSet(best_ind, training_instances, n_test_instances, test_base_idx,
                                features_len, window_size):
    # Transform training instances to feature space
    kNClasses = len(training_instances)
    kNTrainInstances = len(training_instances[1])


    # Compute training features: 1NN base of classification
    train_set = fc.InstancesFeatures(kNClasses*kNTrainInstances, features_len)
    train_set.populate(best_ind, training_instances, window_size)
        
    # Create set of test instances 
    test_instances = []
    for t_i in training_instances:
        test_instances = test_instances + [(t_i[0], range(test_base_idx, test_base_idx + n_test_instances))]

    # Compute the test features
    test_set = fc.InstancesFeatures(kNClasses*n_test_instances, features_len)
    test_set.populate(best_ind, test_instances, window_size)

    # 1NN classification 
    train_matrix = train_set.featuresMatrix()
    dists = np.zeros(shape=train_set.numInstances())
    for i in range(0, test_set.numInstances()):
        for j in range(0, train_set.numInstances()):
            dists[j] = np.linalg.norm(test_set.featuresVector(i) - train_matrix[:,j])
        
        min_idx = np.argmin(dists)
        label_idx = min_idx / kNTrainInstances
        test_set.labelInstance(i, training_instances[label_idx][0])

    return test_set.correctClassifications()[1]

    

#>PARAMETERS--------------------------------------------
#>Statistics parameters
kNRoundsClasses =               1
kNRoundsInstances =             15           #15
#>Training parameters
kNClassesDataset =              112
kNTrainingClasses =             10           #10
kClassesSize =                  100
kNTrainingInstances =           2
#>Algorithm parameters:
kCodeSize =                     5           #5-7
kWindowSize =                   5           #5
#>Evolution parameters:
kPopSize =                      15           #25
kXOverRate =                    0.8
kMutRate =                      0.2
kElitRate =                     0.01
kMaxGenerations =               10           #30
#-------------------------------------------------------

classes = range(1, kNClassesDataset+1)
del classes[14] #dataset problem

#>Iterate over classes
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
    pset = gpc.CreatePrimitiveSet(kWindowSize, kCodeSize)
    tbox = gpc.DefineEvolutionToolbox(pset, sample_instances, kCodeSize, kWindowSize)

    #>Define Log structure
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    #>RERUN THE ALGORITHM WITH DIFFERENT SEEDS
    accs = np.zeros(shape=kNRoundsInstances) 
    for n_in in range(0, kNRoundsInstances):
        print '\nIteration #: ', n_in
        # Generate population
        pop = tbox.generate_population(kPopSize)
        hof = tools.HallOfFame(1)

        # Evolutionary algorithm call
        pop, log = algorithms.eaSimple(pop, tbox, kXOverRate, kMutRate, kMaxGenerations,
                                    stats=mstats, halloffame=hof, verbose=True)
        # ACTIVATE TRY CATCH BLOCK AFTER DEBUG IS COMPLETE (AVOID LIBRARY STABILITY ISSUES)
        # try: 
        #     pop, log = algorithms.eaSimple(pop, tbox, kXOverRate, kMutRate, kMaxGenerations,
        #                                 stats=mstats, halloffame=hof, verbose=True)
        # except:
        #     print 'The evolutionary algorithm failed to evolve'
        #     continue

        print '\nBest individual:\n', hof[0]

        # fh_log = open('./../results/' + str(n_cl) + '_' + str(n_in) + '.log', 'w')
        # fh_hof = open('./../results/' + str(n_cl) + '_' + str(n_in) + '.hof', 'w')
        # pickle.dump(log, fh_log)
        # pickle.dump(hof[0], fh_hof)
        # fh_log.close()
        # fh_hof.close()

        ind_lambda = tbox.compile(hof[0])
        accs[n_in] = ComputeAccuracyOverTestSet(ind_lambda, sample_instances, 10, kClassesSize/2, 2**kCodeSize, kWindowSize)
        print 'Accuracy = ' + accs[n_in]

    print 'Accuracies = ', accs
    print 'Mean = ', np.mean(accs)
    print 'Std Dev = ', np.std(accs)
        
