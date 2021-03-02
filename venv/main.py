import csv
import random
import sys


class Item(object):
    def __init__(self, w, s, c):
        self.weight = w
        self.size = s
        self.cost = c


class Task:
    def __init__(self, n, w, s):
        self.n = n  # number of objects to choose
        self.w = w  # maximum carrying capacity of the knapsack (int)
        self.s = s  # maximum knapsack size
        self.items = []


class Individual(object):
    def __init__(self, genome, task):
        self.genome = genome
        self.fitness = self.evaluate(task)

    def evaluate(self, task):
        k = 0
        weight_sum = 0
        size_sum = 0
        cost_sum = 0

        for i in task.items:
            if self.genome[k] == 1:
                weight_sum += i.weight
                size_sum += i.size
                cost_sum += i.cost
            k = k + 1

        if (weight_sum > task.w or size_sum > task.s):
            return 0
        else:
            return cost_sum


class Population(object):
    def __init__(self, size):
        self.size = size
        self.individuals = []


def generate(n, w, s, output_file):
    f = open(output_file, "w", newline='')
    writer = csv.writer(f)
    writer.writerow([n, w, s])

    w_iSum = 0
    s_iSum = 0

    while not ((w_iSum > 2 * w) and (s_iSum > 2 * s)):

        for i in range(n):
            w_i = random.randint(1, round(10 * w / n))
            s_i = random.randint(1, round(10 * s / n))
            c_i = random.randint(1, n)

            w_iSum += w_i
            s_iSum += s_i
            writer.writerow([w_i, s_i, c_i])

    f.close()
    return


def read(input_file):
    f = open(input_file, "r")
    reader = csv.reader(f, delimiter=",")

    firstrow = True
    for row in reader:

        if (firstrow):
            task = Task(int(row[0]), int(row[1]), int(row[2]))
            firstrow = False
            continue

        item = Item(int(row[0]), int(row[1]), int(row[2]))
        task.items.append(item)
    return task


def init_population(n_items, size, task):
    population = Population(size)

    for i in range(size):
        genome = []

        # Create each gene (bit) randomly
        for j in range(n_items):
            gene = random.choices((0, 1), weights=[5, 1])[0]
            genome.append(gene)

        # Add the genotype to the population
        population.individuals.append(Individual(genome, task))

    return population


def tournament(population, tournament_size):
    best = None
    for i in range(tournament_size):
        contestant = population.individuals[random.randint(0, population.size - 1)]
        if (best == None) or contestant.fitness > best.fitness:
            best = contestant
    return best


def crossover(parent1, parent2, crossover_rate):
    prob = random.uniform(0, 1)
    if (crossover_rate > prob):
        crossover_index = random.randint(1, len(parent1.genome) - 1)
        child_gen = []

        # Join father's genes until crossover point
        for j in range(0, crossover_index):
            child_gen.append(parent1.genome[j])

        # Join mother's genes after crossover point
        for k in range(crossover_index, len(parent2.genome)):
            child_gen.append(parent2.genome[k])

        child = Individual(child_gen, task)
        return child
    else:
        child = parent1
        return child


def mutate(individual, mutation_rate):
    n = round(mutation_rate * len(individual.genome))
    # Select a single gene to mutate
    for x in range(n):
        gene_index = random.randint(0, len(individual.genome) - 1)
        # Flip a single bit
        individual.genome[gene_index] = int(not individual.genome[gene_index])
    return


def popinfo(population):
    fit = 0
    unfit = 0
    fittest = 0
    count = 1
    for x in population.individuals:
        if x.fitness != 0:
            fit = fit + 1
            if x.fitness > fittest:
                fittest = x.fitness
        else:
            unfit = unfit + 1

    print (fit, "fitted")
    print (unfit, "unfitted")
    print (fittest, "fittest")
    print("---")
    return fittest


####################main#####################

random.seed(377)

ITERATIONS = 10
POP_SIZE = 500
CROSSOVER_RATE = 1
MUTATION_RATE = 0.001
TOURNAMENT_SIZE = 250

n = random.randint(1000, 2000)
w = random.randint(10000, 20000)
s = random.randint(10000, 20000)

generate(n, w, s, "knapsack.csv")
task = read("knapsack.csv")
population = init_population(task.n, POP_SIZE, task)

i = 0
while i < ITERATIONS:
    fittest = 0
    comp = popinfo(population)
    if (comp > fittest): fittest = comp
    new_pop = Population(POP_SIZE)
    j = 0
    while j < new_pop.size:
        parent1 = tournament(population, TOURNAMENT_SIZE)
        parent2 = tournament(population, TOURNAMENT_SIZE)
        child = crossover(parent1, parent2, CROSSOVER_RATE)
        mutate(child, MUTATION_RATE)
        new_pop.individuals.append(child)
        j += 1
    population = new_pop
    i += 1

print(fittest, "fittest of all")
