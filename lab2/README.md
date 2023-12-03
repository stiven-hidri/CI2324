There are 3 agents:
 - an agent that plays randomly
 - an agent that plays optimally (makes moves that give nim sum = 0)
 - an agent that is trained using an EA.  
 The strategy used is the following: a population of individuals is created. The genotype for each individual is all the possible states of tha game with the associated move (initally randomly generated).  
 To each individual a fitness score is associated.This is given by the following formula: f = #Moves_With_Resulting_NimSum_0 * 10 + #Items_removed. The more the optimal moves the higher is the fitness and a little bonus is given the more items you take in order to promote faster wins. This bonus must be of a smaller magnitude wrt to the bonus given by the number of optimal moves since if not so we would have final individuals that just take the highest possible number of items per turn.  
 After some generations, mutions, crossovers the most promising individual is returned and he will be the chosen player. His strategy is simple: its strategy function simply takes in input the state of the board and returns the associated move

Some possible optimizations:
- Do not generate equivalent states: (0, 1, 1) is equivalent to (1, 0, 1). It will decrease the number of computations
- Smarter crossover algorithm: instead of mixing random genes mix random promising genes in order to return an agent the is at least ...