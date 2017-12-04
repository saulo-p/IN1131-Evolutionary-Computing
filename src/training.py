#==================================================================================================
# Training/Evolution script
#
# Date created: 4/12/2017
# Code author: Saulo Pereira {scrps@cin.ufpe.br}
#==================================================================================================

import random

import gpcriptor as gpc

#>SCRIPT:------------------------------------------------------------------------------
#>Training parameters
kNClasses = 112
kSizeClass = 100
kNTrainingClasses = 20
kNTrainingInstances = 2
#>Algorithm parameters:
kCS = 2
kWS = 3

#>??
classes = range(1, kNClasses+1)
del classes[14] #dataset problem

# Randomize classes to use on experiment
sample_classes = random.sample(classes, kNTrainingClasses)
sample_classes.sort()

# For each selected class, randomize the training instances
classes_samples = []
for i in sample_classes:
    classes_samples = classes_samples + [(i, random.sample(range(0, kSizeClass), kNTrainingInstances))]
print classes_samples

#>Evolution framework
