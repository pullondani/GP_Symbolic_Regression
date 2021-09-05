import random
import operator
import numpy
import csv
import math
from sympy import sympify
from deap import algorithms, base, creator, tools, gp


def readFile():
    xVals = []
    yVals = []
    with open('data.txt', 'r') as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            xVals.append(float(row[0]))
            yVals.append(float(row[1]))

        return xVals, yVals


xVals, yVals = readFile()


def evaluate(individual, points):
    func = toolbox.compile(expr=individual)
    err = 0.
    for i, x in enumerate(points):
        err += abs(func(x) - yVals[i])
    return err / len(points),


def protectedDiv(left, right):
    if (right == 0):
        return 1.
    else:
        try:
            return left / right
        except ZeroDivisionError:
            return 1.


pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.sin, 1)
# pset.addPrimitive(operator.pow, 2)

# pset.addTerminal(2.)
pset.addTerminal(3.)

pset.renameArguments(ARG0="x")

pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evaluate, points=xVals)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=2, max_=6)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(
    key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(
    key=operator.attrgetter("height"), max_value=17))

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)


locals = {
    'sub': lambda x, y: x - y,
    'protectedDiv': protectedDiv,
    'mul': lambda x, y: x*y,
    'add': lambda x, y: x + y,
    'neg': lambda x: -x
}


def main():
    # random.seed(84)

    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    CXPB, MUTPB, NGEN = 0.9, 0.1, 50
    pop, log = algorithms.eaSimple(
        pop, toolbox, CXPB, MUTPB, NGEN, stats=mstats, halloffame=hof, verbose=True)
    frmt = sympify(str(hof[0]), locals=locals)
    print(hof[0])
    print(f'simplified: {frmt}')
    print('Average Error Per Input', evaluate(hof[0], points=xVals)[0])

    return pop, log, hof


if __name__ == '__main__':
    main()
