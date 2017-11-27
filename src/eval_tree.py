import operator
from deap import base, creator, gp, tools

#>Step 0: Primitive set (loosely typed) definition
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addTerminal(1)
pset.addTerminal(0.1)
pset.renameArguments(ARG0='x')

# #>TEST: Print randomly generated tree (from defined primitive set)
# expr = gp.genFull(pset, min_=1, max_=3)
# tree = gp.PrimitiveTree(expr)
# print tree
# #------------------------------------------

#>Step 1: Create classes for individuals and fitness (optional in some cases)
#   "create" defines the new class (string) that inherits from second argument
#creator.create("Fitness", base.Fitness, weigths=-1.0)
#creator.create("Individual", gp.PrimitiveTree)

#>Step 2: Create and populate evolution toolbox
tbox = base.Toolbox()
tbox.register("generate_expr", gp.genFull, pset, min_=2, max_=5)
tbox.register("generate_tree", tools.initIterate, gp.PrimitiveTree, 
              tbox.generate_expr)

# #>TEST: Print individual generated from toolbox interface
# s = tbox.generate_tree()
# print tbox.generate_tree()
# #------------------------------------------

#>Step 3: Compile expression/tree (partial evaluation)
expr = tbox.generate_tree()
print expr
# print "\n\n"
eval = gp.compile(expr, pset)

#>Step 4: Complete evaluation
print "\nResult:", eval(1)