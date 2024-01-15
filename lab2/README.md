## optimal and random agent

```py
class Agent:
    _name = "AGENT"

    def __str__(self):
        return f"{self._name}"

    def nim_sum(self, game: Game) -> int: #nim sum
        if game.K is None:
            source = game.Rows #game rows
        else:
            source = [x % (game.K+1) for x in game.Rows] #mex
        
        tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in source])
        xor = tmp.sum(axis=0) % 2
        return int("".join(str(_) for _ in xor), base=2)

    def get_possible_moves(self, game: Game) -> dict:
        possible_moves = dict()
        for move in (Move(r, o) for r, c in enumerate(game.Rows) for o in range(1, c + 1)):
            if game.K is not None and move.taken > game.K:
                continue    
            
            game_aftermove = deepcopy(game)
            game_aftermove.action(move)
            possible_moves[move] = self.nim_sum(game_aftermove) #for each move (row, taken) i save the resulting nim sum

        return possible_moves

class randomAgent(Agent):
    _name = "RND"
    
    def strategy(self, game: Game) -> Move:
        possible_moves = super().get_possible_moves(game)
        return random.choice(list(possible_moves.keys()))
    
class optimalAgent(Agent):
    _name = "OPT"

    def strategy(self, game: Game) -> Move:
        move = None

        possible_moves = Agent.get_possible_moves(self, game)
        optimal_moves = [move for move, nim_sum in possible_moves.items() if nim_sum == 0]
        if len(optimal_moves)>0:
            move = random.choice(optimal_moves)
        else:
            move = random.choice(list(possible_moves.keys()))

        return move
```

## EA AGENT
Adopted strategy:
 1. a population of individuals is created. The genotype consists of all the possible states of the game with an associated move (randomly generated).
 2. To each individual a fitness score is associated. This is given by the following formula: f = #Moves_With_Resulting_NimSum_0 * 10 + #Items_removed. A little bonus is given the more items you take in order to promote faster wins. This bonus must be of a smaller magnitude wrt to the bonus given by the number of optimal moves since if not so we would have final individuals that just take the highest possible number of items per turn.  
 3. After some generations, mutations and crossovers the most promising individual is returned

```py
class eaAgent(Agent):
    _POPULATION_SIZE = 1000
    _OFFSPRING_SIZE = 100
    _MUTATION_PROBABILITY = .3
    _TOURNAMENT_SIZE = 10
    _GENERATIONS = 200

    def __init__(self, N, k) -> None:
        self._N = N
        self._k = k
        self._name = "EA"
        self._std_genotype = self.generate_genotype(N)
        self.strongest = self.train()
        
    def generate_genotype(self, N):
        sets = []
        for i in range(N):
            # Each set has numbers from 0 to i*2+1
            sets.append(list(range(i * 2 + 2)))

        # Use itertools.product to generate all combinations
        all_combinations = list(product(*sets))
        all_states = [Field(c, self._k) for c in all_combinations if sum(c)>0]
        genes = dict()

        for s in all_states:
            genes[s]=None

        return genes

    def nim_sum(self, gene:tuple[Field, Move]) -> int: #nim sum
        field = deepcopy(gene[0])
        field.nimming(gene[1])

        if field.K is None:
            source = field.Rows #game rows
        else:
            source = [x % (field.K+1) for x in field.Rows] #mex
        
        tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in source])
        xor = tmp.sum(axis=0) % 2
        return int("".join(str(_) for _ in xor), base=2)

    def initialize_genes(self): 
        geno = deepcopy(self._std_genotype)
        for g in geno:
            target = random.choice([(i,r) for i, r in enumerate(g._rows) if r>0])
            items_to_take = random.randint(1, min([target[1], self._k]))
            assert(g._rows[target[0]]>=items_to_take)
            geno[g]=Move(target[0], items_to_take)

        return geno

    @dataclass
    class Individual:
        fitness: int
        genotype: dict


    def fitness(self, genotype:dict[Field, Move]):
        score = 0
        for k,v in genotype.items():
            score += int(self.nim_sum((k,v))==0)*10 + v.taken

        return score

    def select_parent(self, pop):
        pool = random.sample(pop, self._TOURNAMENT_SIZE)
        champion = max(pool, key=lambda i: i.fitness)
        return champion

    def mutate(self, ind: Individual) -> Individual:
        offspring = deepcopy(ind)
        key = random.choice([key for key in ind.genotype.keys()])
        
        target = random.choice([(i,r) for i, r in enumerate(key._rows) if r>0])
        items_to_take = random.randint(1, min([target[1], self._k]))

        assert(key._rows[target[0]]>=items_to_take)
        offspring.genotype[key]=Move(target[0], items_to_take)

        offspring.fitness = self.fitness(offspring.genotype)
        return offspring

    def one_cut_xover(self, ind1: Individual, ind2: Individual) -> Individual:
        incoming_genes = random.randint(1, len(ind1.genotype.keys()))
        crossing_genes = random.sample([key for key in ind1.genotype.keys()], k=incoming_genes)
        offspring = self.Individual(fitness=0, genotype=deepcopy(ind1.genotype))
        for k in crossing_genes:
            offspring.genotype[k]=deepcopy(ind2.genotype[k])
            
        offspring.fitness=self.fitness(offspring.genotype)

        return offspring

    def train(self) -> Individual:
        population = [ self.Individual(genotype=self.initialize_genes(), fitness=0) for _ in range(self._POPULATION_SIZE) ]

        for p in population:
            p.fitness = self.fitness(p.genotype)

        print(f"Generations: {self._GENERATIONS}")
        print("Training EA...")
        for gen in range(self._GENERATIONS):
            print(f"\rG{gen+1}", end='', flush=True)
            offspring = []
            for counter in range(self._OFFSPRING_SIZE):
                if random.random() < self._MUTATION_PROBABILITY:
                    # mutation
                    p = self.select_parent(population)
                    o = self.mutate(p)
                else:
                    # xover
                    p1 = self.select_parent(population)
                    p2 = self.select_parent(population)
                    o = self.one_cut_xover(p1, p2)
                offspring.append(o)

            population.extend(offspring)
            population.sort(key=lambda i: i.fitness, reverse=True)
            population = population[:self._POPULATION_SIZE]

        print("\nDone...")

        return max(population, key=lambda i: i.fitness)
    
    def strategy(self, game:Game) -> Move:
        return self.strongest.genotype[game.field]
```