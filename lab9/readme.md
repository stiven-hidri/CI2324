In order to write a local-search algorithm able to solve the *Problem* instances 1, 2, 5, and 10 using a minimum number of fitness calls I adopted the following strategies:

**Mutation:** randomly mutate 1 gene
```py
def mutate(ind: Individual) -> Individual:
    offspring = deepcopy(ind)
    pos = randint(0, K-1)
    offspring.genotype[pos] = 1 - offspring.genotype[pos]
    offspring.fitness = -1
    return offspring
```
**Reorder:** shuffle genes
```py
def reorder(ind: Individual) -> Individual:
    offspring = deepcopy(ind)
    shuffle(offspring.genotype)
    return offspring
```
**xover:** randomly crossover N genes with another individual, with N randomly generated
```py
def xover(ind1: Individual, ind2: Individual) -> Individual:
    how_many = randint(1, K)
    targets = sample(range(K), k=how_many)
    offspring = deepcopy(ind1)
    for t in targets:
        offspring[t] = ind2[t] 
    assert len(offspring.genotype) == K
    return offspring
```
Here is presented the **main code**:
```py
for s in PROBLEM_SIZE:
    fitness = make_problem(s)
    pop = deepcopy(population)
    maxF = -1
    result = "Failure"
    
    for p in pop:
        p.fitness = fitness(p)
        if p.fitness>maxF:
            maxF = p.fitness
        
    g=0
    stillNoPerfectIndividual = True
    while g < GENERATIONS and stillNoPerfectIndividual:
        offspring = list()
        for counter in range(OFFSPRING_SIZE):
            o = None
            
            if random() < REORDER_PROBABILITY:
                if o is not None:
                    o = reorder(o)
                else:
                    p = select_parent(pop)
                    o = reorder(p)
            
            if random() < MUTATION_PROBABILITY:
                if o is not None:
                    o = mutate(o)
                else:
                    p = select_parent(pop)
                    o = mutate(p)
                    
            if random() < XOVER_PROBABILITY:
                # xover # add more xovers
                if o is not None:
                    p2 = select_parent(pop)
                    o = xover(o, p2)
                else:
                    p1 = select_parent(pop)
                    p2 = select_parent(pop)
                    o = xover(p1, p2)

            if o is not None:
              offspring.append(o)

        for o in offspring:
            o.fitness = fitness(o.genotype)

            if o.fitness>maxF:
                maxF = o.fitness
            
            if maxF == 1.0:
                stillNoPerfectIndividual=False
                result = "Success"
                break

        if not stillNoPerfectIndividual:
            break

        pop.extend(offspring)
        pop.sort(key=lambda i: i.fitness, reverse=True)
        pop = pop[:POPULATION_SIZE]
    
        g+=1

        print(f"S={s}\t{round(g/GENERATIONS*100, 2)}%\t\tmaxFitness:{round(maxF,2)}\t\tcalls:{fitness.calls}", end="\r")

    print(f"S={s}\t\tresult:{result}\t\tmaxFitness:{round(maxF,2)}\t\tcalls:{fitness.calls}", end="\n")
```

**parameters:** maximum probability for xover, mutation and reordering led to the best results
```py
POPULATION_SIZE = 300
OFFSPRING_SIZE = 30
MUTATION_PROBABILITY = 1
XOVER_PROBABILITY = 1
REORDER_PROBABILITY = 1
TOURNAMENT_SIZE = 300
GENERATIONS = 1000
PROBLEM_SIZE = [1,2,5,10]
K=1000
```

## RESULT
| Problem   | Result            | maxFitness        | Calls         |
|:---------:|:-----------------:|:-----------------:|:-------------:|
| S=1       | Success           | 1.0               | 1217          |
| S=2       | Success           | 1.0               | 6302          |
| S=5       | Failure           | 0.6               | 30010         |
| S=10      | Failure           | 0.38              | 30010         |
