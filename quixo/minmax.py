from random import choice as rndchoice, shuffle as rndshuffle
from numpy import sum as npsum, diag as npdiag, fliplr as npfliplr, diag as npdiag
from itertools import product as itproduct
from copy import copy
from game import Game, Move, Player

BOARD_SIZE = 5

pool_moves = [
    *itproduct([(0,1),(0,2),(0,3)], [Move.BOTTOM, Move.LEFT, Move.RIGHT]), 
    *itproduct([(4,1),(4,2),(4,3)], [Move.LEFT, Move.RIGHT, Move.TOP]), 
    *itproduct( [(1,0),(2,0),(3,0)],  [Move.BOTTOM, Move.TOP, Move.RIGHT]), 
    *itproduct([(1,4),(2,4),(3,4)], [Move.BOTTOM, Move.LEFT, Move.TOP]), 
    ((0,0),Move.BOTTOM), ((0,0),Move.RIGHT), ((0,4),Move.BOTTOM), ((0,4),Move.LEFT), ((4,0),Move.TOP), ((4,0),Move.RIGHT), ((4,4),Move.TOP), ((4,4),Move.LEFT)
]

rndshuffle(pool_moves)

time_per_move = []

class RandomPlayer(Player):
    def __init__(self, player_id = 1) -> None:
        super().__init__()
        self.player_id = player_id
        self.last_state_action = None
        self.all_moves = pool_moves

    def get_valid_moves(self, state):
        return [ x for x in self.all_moves if state[x[0]] == -1 or state[x[0]] == self.player_id ]

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:        
        from_pos, slide = rndchoice(self.get_valid_moves(game._board))
        return tuple(reversed(from_pos)), slide

class MinMaxAgent(Player):
    class Node:
        def __init__(self, board, move):
            self.board = board
            self.move = move

    def __init__(self, depth=2, player_id=0):
        super().__init__()
        self.depth = depth
        self.player_id = player_id
        self.all_moves = pool_moves
        self.start = 1

    def get_valid_moves(self, board, player_id):
        return [x for x in self.all_moves if board[x[0]] == -1 or board[x[0]] == player_id]
    
    def get_first_move(self, board, player_id):
        for x in self.all_moves:
            if board[x[0]] == -1 or board[x[0]] == player_id:
                return x

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

    def check_if_winner(self, board, opponent):
        # Check rows
        for row in board:
            if npsum(row == opponent) == BOARD_SIZE:
                return True

        # Check columns
        for col in board.T:
            if npsum(col == opponent) == BOARD_SIZE:
                return True

        # Check main diagonal
        if npsum(npdiag(board) == opponent) == BOARD_SIZE:
            return True

        # Check anti-diagonal
        if npsum(npdiag(npfliplr(board)) == opponent) == BOARD_SIZE:
            return True
    
        return False

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
        
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:

        if self.start==1:
            optimal_move = self.get_first_move(game._board, self.player_id)
            self.start=0
        else:
            root = self.Node(game._board, None)
            _, optimal_move = self.minimax(root, self.depth, float('-inf'), float('inf'), True)

        return tuple(reversed(optimal_move[0])), optimal_move[1]

def main():
    game = Game()

    myAgent = MinMaxAgent(player_id=0, depth=3)
    opponent = RandomPlayer(player_id=1)
    
    result = game.play(myAgent, opponent)

    print(f"WINNER: {result}")

if __name__ == '__main__':
    main()