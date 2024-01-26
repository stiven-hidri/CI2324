## Q-Learning - 1st attempt
Since i have worked on q_learning for the tictactoe lab I decided to try it on quixo.  
Ufortunately I kept getting very modest results vs the random player (around 55% of winnings) so I tried to apply some heuristics in order to help the agent to choose the best move: give a bonus when 4 consecutive pieces have been placed, when a 4-consecutive piece of the opponent is disrupted and give a score to each single state in order to promote better moves but still no improvement at all.
I sterted to think that maybe the low performances were due to the enourmous number of possibe states of the game.
So I last tried to take into consideration also equivalent states (rotations of the board, flipping u/d and l/r and inverting the pieces) but still no significant boost of performances.
So I decided to change completely the approach and I opted for the minimax algorithm with the alpha beta pruning optimization.
I wasn't too concerned to change strategy since minimax is relatively fast to implement and I could bring the heuristics I studied when implementing Q-Learning into the new model.
I left the code inside the folder "other_attempts" in order to give an idea to the work I have done. It shouldn't run since there have been updates to the game.py file and I have not kept the code udpated.
## Minimax with Alpha Beta pruning
I immediately got very high results with a first version versus the random player and so I decided to keep going and improve the performances of the code.

## Key points
**Evaluation**: Here there is the code of the evaluation function:

```python
def evaluate(self, board):
        def get_score(id):
            
            base = 3
            score = 0

            boardt = board.T
            boardf = npfliplr(board)

            #rows and columns
            r0, rt0, r1, rt1, r2, rt2, r3, rt3, r4, rt4 = board[0], boardt[0], board[1], boardt[1], board[2], boardt[2], board[3], boardt[3], board[4], boardt[4]
            #main and second diagonals
            md = [board[0][0], board[1][1], board[2][2], board[3][3], board[4][4]]
            sd = [boardf[0][0], boardf[1][1], boardf[2][2], boardf[3][3], boardf[4][4]]

            #rows
            factor = npsum(r0 == id)
            score += factor*base**factor
            factor = npsum(r1 == id)
            score += factor*base**factor
            factor = npsum(r2 == id)
            score += factor*base**factor
            factor = npsum(r3 == id)
            score += factor*base**factor
            factor = npsum(r4 == id)
            score += factor*base**factor

            #columns
            factor = npsum(rt0 == id)
            score += factor*base**factor
            factor = npsum(rt1 == id)
            score += factor*base**factor
            factor = npsum(rt2 == id)
            score += factor*base**factor
            factor = npsum(rt3 == id)
            score += factor*base**factor
            factor = npsum(rt4 == id)
            score += factor*base**factor

            #main diagonal
            factor = npsum(md == id)
            score += factor*base**factor

            #diagonale secondaria
            factor = npsum(sd == id)
            score += factor*base**factor

            return score
        
        my_bonus, opp_bonus = get_score(self.player_id), get_score(1-self.player_id)
        
        return my_bonus-opp_bonus 
```
The idea is to give a bonus for each group of pieces owned in rows, columns and diagonals which exponentially increases with the number of pieces. Doing so promotes having more pieces in the board but mainly reaching 5 consecutive pieces. We calculate with the same strategy also the score of the opponent and we subtruct it to the score of the player. In this way we promote winning and keeping the opponent in a loosing state.
The code isn't scalable but I written it in this way since I tried to make it faster since this function will bee called many times

**minimax algorithm**: here we can see the implementation of the minimax algoirthm. I tried various settings and the following proved to be the most efficient.
The terminal conditions for the recursive functions are either one of the 2 players won or the maximum depth is reached.
I iterate through the moves and generate the state that comes with the current move: doing so i avoid generating states (which is computationally expensive) that i don't need to generate when i'm not traversing some subtree thanks to alpha beta pruning. The minimax function returns both with the state value the related move. In this way it immediately and elegantely returns the action to take at the end.
If it is the first time playing we just make a move.

```py
def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:

        if self.start==1:
            optimal_move = self.get_first_move(game._board, self.player_id)
            self.start=0
        else:
            root = self.Node(game._board, None)
            _, optimal_move = self.minimax(root, self.depth, float('-inf'), float('inf'), True)

        return tuple(reversed(optimal_move[0])), optimal_move[1]

def minimax(self, node, depth, alpha, beta, maximizing):
        # check if opponent won
        if self.check_if_winner(node.board, 1-self.player_id):
            return float("-inf"), node.move

        # check if I won
        if self.check_if_winner(node.board, self.player_id):  
            return float("inf"), node.move

        # when max depth is reached return the score obtained with corresponding move
        if depth == 0:
            return self.evaluate(node.board), node.move

        # if maximizing the current player is myself, if not the current player is the opponent
        player_id = self.player_id if maximizing else 1-self.player_id

        proto_game = Game() 
        moves = self.get_valid_moves(node.board, player_id)

        best_eval = float('-inf') if maximizing else float('inf')
        optimal_move = node.move

        for m in moves:
            proto_game._board = copy(node.board)
            proto_game._Game__move(tuple(reversed(m[0])), m[1], player_id)
            child = self.Node(proto_game._board, m)

            eval, _ = self.minimax(child, depth - 1, alpha, beta, not maximizing)

            if maximizing and eval > best_eval:
                optimal_move = child.move
                best_eval = eval

                alpha = max(alpha, best_eval)
            elif (not maximizing) and eval < best_eval:
                best_eval = eval
                optimal_move = child.move

                beta = min(beta, best_eval)
            
            if beta <= alpha:
                break

        return (best_eval, optimal_move)


```

**get_valid_moves:** in order to get the valid moves i used the following lines of code:

```py
pool_moves = [
    *itproduct([(0,1),(0,2),(0,3)], [Move.BOTTOM, Move.LEFT, Move.RIGHT]), 
    *itproduct([(4,1),(4,2),(4,3)], [Move.LEFT, Move.RIGHT, Move.TOP]), 
    *itproduct( [(1,0),(2,0),(3,0)],  [Move.BOTTOM, Move.TOP, Move.RIGHT]), 
    *itproduct([(1,4),(2,4),(3,4)], [Move.BOTTOM, Move.LEFT, Move.TOP]), 
    ((0,0),Move.BOTTOM), ((0,0),Move.RIGHT), ((0,4),Move.BOTTOM), ((0,4),Move.LEFT), ((4,0),Move.TOP), ((4,0),Move.RIGHT), ((4,4),Move.TOP), ((4,4),Move.LEFT)
]

def get_valid_moves(self, board, player_id):
    return [x for x in pool_moves if board[x[0]] == -1 or board[x[0]] == player_id]
```

**pool_moves** contains all the possible moves one can make (44 in total) and then with **get_valid_moves** we get the moves that we can do given the board sate and our player_id
