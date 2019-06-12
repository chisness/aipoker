from math import *
import numpy as np
import copy
import random
import math

#minimax, negamax
#alpha beta
#mcts
#random from winning/drawing/losing moves
#mega tictactoe

class Tictactoe:
	def __init__(self, board = [0] * 9, acting_player = 1):
		self.board = board
		self.acting_player = acting_player

	def make_move(self, move):
		if move in self.available_moves():
			self.board[move] = self.acting_player
			self.acting_player = 0 - self.acting_player #players are 1 or -1

	def new_state_with_move(self, move): #return new ttt state with move, but don't change this state
		if move in self.available_moves():
			board_copy = copy.deepcopy(self.board)
			board_copy[move] = self.acting_player
			return Tictactoe(board_copy, 0 - self.acting_player)
			
	def available_moves(self):
		return [i for i in range(9) if self.board[i] == 0]
	
	def check_result(self):
		for (a,b,c) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
			if self.board[a] == self.board[b] == self.board[c] != 0:
				return self.board[a]
		if self.available_moves() == []: return 0 #Tie
		return None #Game not over
		
	def __repr__(self):
		s= ""
		for i in range(9): 
			if self.board[i] == 0:
				s+=str(i)
			elif self.board[i] == 1:
				s+='x'
			elif self.board[i] == -1:
				s+='o'
			if i == 2 or i == 5: s += "\n"
		return s


class NegamaxAgent:
	def __init__(self):
		self.memo = {} #move, value

	def negamax(self, game_state):
		if game_state not in self.memo: #already visited this state?
			result = game_state.check_result()
			if result is not None: #leaf node or end of search
				best_move = None
				best_val = result * game_state.acting_player #return 0 for tie or 1 for maximizing win or -1 for minimizing win
			else:
				best_val = float('-inf')
				for i in game_state.available_moves():
					clone_state = copy.deepcopy(game_state)
					clone_state.make_move(i) #makes move and switches to next player
					_, val = self.minimax(clone_state)
					val *= -1 
					if val > best_val:
						best_move = i
						best_val = val	
			self.memo[game_state] = (best_move, best_val)
		return self.memo[game_state]


class MCTSNode:
	def __init__(self, game_state, parent = None, move = None):
		self.parent = parent
		self.move = move
		self.game_state = game_state
		self.children = []
		self.win_counts = {1: 0, -1: 0}
		self.num_rollouts = 0
		self.unvisited_moves = game_state.available_moves()

	def add_random_child(self):
		move_index = random.randint(0, len(self.unvisited_moves)-1) #inclusive
		new_move = self.unvisited_moves.pop(move_index)
		new_node = MCTSNode(self.game_state.new_state_with_move(new_move), self, new_move)
		self.children.append(new_node)
		return new_node

	def can_add_child(self):
		return len(self.unvisited_moves) > 0

	def is_terminal(self):
		return self.game_state.check_result() is not None

	def update(self, result):
		if result == 1:
			self.win_counts[1] += 1
			self.win_counts[-1] -= 1
		elif result == -1:
			self.win_counts[-1] += 1
			self.win_counts[1] -= 1
		self.num_rollouts += 1

	def winning_frac(self, player):
		return float(self.win_counts[player]) / float(self.num_rollouts)


class MCTSAgent:
	def __init__(self, num_rounds = 10000, temperature = 2):
		self.num_rounds = num_rounds
		self.temperature = temperature

	def uct_select_child(self, node):
		best_score = -float('inf')
		best_child = None
		total_rollouts = sum(child.num_rollouts for child in node.children)
		log_rollouts = math.log(total_rollouts)

		for child in node.children:
			win_pct = child.winning_frac(node.game_state.acting_player)
			exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
			uct_score = win_pct + self.temperature * exploration_factor
			if uct_score > best_score:
				best_score = uct_score
				best_child = child
		return best_child

	def select_move(self, game_state):
		root = MCTSNode(game_state)

		for i in range(self.num_rounds):
			node = root
			print(i)
			print(node.game_state)

			#selection -- UCT select child until we get to a node that can be expanded
			while (not node.can_add_child()) and (not node.is_terminal()):
				node = self.uct_select_child(node)
				print(node.game_state)
				print('selection')

			#expansion -- expand from leaf unless leaf is end of game
			if node.can_add_child():
				node = node.add_random_child()
				print('expansion')
				print(node.game_state)

			#simulation -- complete a random playout from the newly expanded node
			gs_temp = copy.deepcopy(node.game_state)
			while gs_temp.check_result() is None:
				print('simulation')
				gs_temp.make_move(random.choice(gs_temp.available_moves()))
				print(gs_temp)

			# while gs.available_moves() != []:
			# 	gs.make_move(random.choice(gs.available_moves()))

			#backpropagation -- update all nodes from the selection to leaf stage
			while node is not None:
				print('backprop')
				print(gs_temp.check_result())
				node.update(gs_temp.check_result())
				node = node.parent

		scored_moves = [(child.winning_frac(game_state.acting_player), child.move, child.num_rollouts) for child in root.children]
		scored_moves.sort(key = lambda x: x[0], reverse=True)
		for s, m, n in scored_moves[:10]:
			print('%s - %.3f (%d)' % (m, s, n))

		best_pct = -1.0
		best_move = None
		for child in root.children:
			child_pct = child.winning_frac(game_state.acting_player)
			if child_pct > best_pct:
				best_pct = child_pct
				best_move = child.move
		print('Select move %s with avg val %.3f' % (best_move, best_pct))
		return best_move


class HumanAgent:
	def select_move(self, game_state):
		print('Enter your move (0-8): ')
		move = int(float(input()))
		#print('move', move)
		#print('game state available moves', game_state.available_moves())
		if move in game_state.available_moves():
			return move
		else:
			print('Invalid move, try again')
			self.select_move(game_state)
	
class RandomAgent:
	def select_move(self, game_state):
		return random.choice(game_state.available_moves())

if __name__ == "__main__":
	#ttt = Tictactoe([0,0,-1,0,0,0,1,-1,1])
	ttt = Tictactoe()
	#ttt = Tictactoe([1,0,0,-1,0,0,1,0,-1])
	#ttt = Tictactoe([1,0,0,0,0,0,0,0,0])
	#ttt = Tictactoe([1,-1,1,1,-1,-1,0,1,0],-1)
	#ttt = Tictactoe([-1,1,-1,-1,1,1,0,-1,0])
	#ttt = Tictactoe([0,0,-1,1,0,0,0,0,0],-1)
	#ttt = Tictactoe([1,0,0,0,0,0,0,0,-1])
	print(ttt)
	#mms = MinimaxAgent()
	#h = HumanAgent()
	mctsa = MCTSAgent(num_rounds = 1000)
	print(mctsa.select_move(ttt))
	#print(mms.minimax(ttt))
	#agent = MinimaxAgent()
	#print(agent.select_move(ttt, 1))
	# moves = 0
	# while ttt.available_moves():
	# 	print(ttt)
	# 	print('\n')
	# 	print('move', moves)
	# 	print('acting player', ttt.acting_player)
	# 	if moves % 2 == 0:
	# 		print(ttt.board)
	# 		move, _ = mms.minimax(ttt)
	# 		print('minimax move', move)
	# 	else:
	# 		move = h.select_move(ttt)
	# 	ttt.make_move(move)
	# 	if ttt.check_result() == 0:
	# 		print('Draw game')
	# 		break
	# 	elif ttt.check_result() == 1:
	# 		print('Player 1 wins')
	# 	elif ttt.check_result() == -1:
	# 		print('Player 2 wins')
	# 	moves+=1





