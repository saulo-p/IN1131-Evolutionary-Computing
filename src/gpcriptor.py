#!/usr/bin/env python
# Image Descriptor: A Genetic Programming Approach to Multiclass Texture Classification
# 
# Date created: 28/11/2017
# Code author: Saulo Pereira {scrps@cin.ufpe.br}
#==================================================================================================

import itertools
import operator
from deap import base, creator, gp, tools

#>Algorithm parameters
# Constants (TODO: use as arguments, not global variables)
kNF = 4
kWS = 3

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

#>Function set (terminals come via evaluation)
pset = gp.PrimitiveSetTyped("GP-criptor", itertools.repeat(float, kWS**2), tuple, "Px")
# Arithmetic operations
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
# Root node
pset.addPrimitive(codeFunction, [float]*kNF, tuple)


#>Evolution toolbox
tbox = base.Toolbox()
tbox.register("generate_expr", gp.genFull, pset, min_=2, max_=3)
tbox.register("generate_tree", tools.initIterate, gp.PrimitiveTree, tbox.generate_expr)

#>TEST: Print individual generated from toolbox interface
expr = tbox.generate_tree()
print expr
peval = gp.compile(expr,pset)
print 'List: ', peval(0,1,2,3,4,5,6,7,8)
#------------------------------------------              