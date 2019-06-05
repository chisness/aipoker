from math import *
import numpy as np
import copy
import random

#minimax, negamax
#alpha beta
#mcts
#mega tictactoe

class Tictactoe:
	def __init__(self, board = [0] * 9, acting_player = 1):
		self.board = board
		self.acting_player = acting_player

	def make_move(self, move):
		if move in self.available_moves():
			self.board[move] = self.acting_player
			self.acting_player = 0 - self.acting_player #players are 1 or -1
			
	def available_moves(self):
		return [i for i in range(9) if self.board[i] == 0]

	def next_states(self):
		a_s = []
		for i in self.available_moves():
			print('access!')
			t = self.board #copy the board
			t[i] = self.acting_player
			a_s.append(Tictactoe(t, 0-self.acting_player))
		return a_s
	
	def check_result(self):
		for (a,b,c) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
			if self.board[a] == self.board[b] == self.board[c] != 0:
				return self.board[a]
		if self.available_moves == []: return 0 #Tie
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

class MinimaxAgent:
	def __init__(self):
		self.memo = {} #state, value
		#returns value, action

	def minimax(self, game_state):#, depth = 12):
		if game_state.acting_player not in self.memo:
			self.memo[player] = {}

		if game_state not in self.memo[player]: #already visited this state?

			result = game_state.check_result()
			if result not None:# or depth == 0: #leaf node or end of search
				best_move = None
				best_val = result #return 0 for tie or 1 for maximizing win or -1 for minimizing win

			if game_state.acting_player == 1: #maximizing node
				v = []
				best_val = float('-inf')
				for i in game_state.available_moves():
					print('available moves max', i)
					t = game_state.board
					t[i] = game_state.acting_player
					new_state = Tictactoe(t, 0-game_state.acting_player)
					print('max state\n',new_state)
					v[i] = self.minimax(new_state)#, depth-1)
					if v[i] > best_val:
						best_val = v[i]
						best_action = i
				return best_val, best_action

			if game_state.acting_player == -1: #minimizing node
				v = []
				best_val = float('inf')
				for i in game_state.available_moves():
					print('available moves min', i)
					t = game_state.board
					t[i] = game_state.acting_player
					new_state = Tictactoe(t, 0-game_state.acting_player)
					print('min state\n',new_state)
					v[i] = self.minimax(new_state)#, depth-1)
					if v[i] < best_val:
						best_val = v[i] 
						best_action = i
			self.memo[player][game_state] = (best_move, best_val)
		return self.memo[player][game_state]


class MinimaxDepthAgent:
	def __init__(self):
		self.memo = {} #state, value

	def select_move(self, game_state, depth = 9):
		if game_state not in self.memo:
			val = game_state.check_result()
			if val is None and depth>=1: #game not over and depth not reached
			#if depth reached, then 
				val = max(self.select_move(possible_state, depth-1) * game_state.acting_player for possible_state in game_state.next_states())
				self.memo[game_state] = val * game_state.acting_player
			else: #game is over
				self.memo[game_state] = val 
		return self.memo[game_state]

class AlphaBetaAgent:
	def __init__(self, alpha, beta):
		self.memo = {} #state, value

	def select_move(self, game_state, alpha = -float('inf'), beta = float('inf')):
		if game_state not in self.memo:
			val = game_state.check_result()
			if val is None: #game not over
				val = max(self.select_move(possible_state, depth-1) * game_state.acting_player for possible_state in game_state.next_states())
				self.memo[game_state] = val * game_state.acting_player
			else: #game is over
				self.memo[game_state] = val 
		return self.memo[game_state]

# class MinimaxAgentR:
# 	def __init__(self):
# 		self.memo = {} #state, value

# 	def select_move(self, game_state, depth = 9):
# 		winning_moves = []
# 		draw_moves = []
# 		losing_moves = []

# 		if game_state not in self.memo:
# 			val = state.check_result()
# 			if val is None: #game not over
# 				for possible_move in game_state.available_moves():	
# 					val = max(self.select_move() * game_state.player())
# 					self.memo[state] = val * game_state.player()
# 			else: #game is over
# 				self.memo[state] = val 
# 		return self.memo[state]

# 		#if depth == 0:
# 		#	return 

# 		# for possible_move in game_state.available_moves():
# 		# 	clone = copy.copy(game_state)
# 		# 	clone.make_move(possible_move)
# 		# 	best_result = self.best_result(clone, player)#, depth-1)

# 		# 	if best_result == player:
# 		# 		winning_moves.append(possible_move)
# 		# 	elif best_result == 0.5:
# 		# 		draw_moves.append(possible_move)
# 		# 	else:
# 		# 		losing_moves.append(possible_move)

# 		# print('winning', winning_moves)
# 		# print('draw', draw_moves)
# 		# print('losing', losing_moves)

# 		# if winning_moves:
# 		# 	return random.choice(winning_moves)
# 		# if draw_moves:
# 		# 	return random.choice(draw_moves)
# 		# return random.choice(losing_moves)


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
	ttt = Tictactoe()
	print(ttt)
	mms = MinimaxAgent()
	print(mms.minimax(ttt))
	#agent = MinimaxAgent()
	#print(agent.select_move(ttt, 1))



