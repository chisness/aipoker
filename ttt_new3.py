from math import *
import numpy as np
import copy
import random

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


class MinimaxAgent:
	def __init__(self):
		self.memo = {} #move, value

	def minimax(self, game_state):
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


class AlphaBetaAgent:
	#Initialize alpha = -inf and beta = inf
	#Prune when alpha >= beta
	#Alpha min score for maximzier
	#Beta max score for minimizer, first minimizer score will = new beta

	def __init__(self):
		self.memo = {} #state, value
		#returns value, action

	def minimax(self, game_state, alpha, beta):
		if game_state not in self.memo: #already visited this state?

			result = game_state.check_result()
			if result is not None: #leaf node or end of search
				best_move = None
				best_val = result * game_state.acting_player #return 0 for tie or 1 for maximizing win or -1 for minimizing win

			else:
				print(game_state)
				best_val = float('-inf')
				for i in game_state.available_moves():
					clone_state = copy.deepcopy(game_state)
					clone_state.make_move(i) #makes move and switches to next player
					print(clone_state)
					_, val = self.minimax(clone_state)
					val *= -1 #$clone_state.acting_player
					if val > best_val:
						best_move = i
						best_val = val	

			self.memo[game_state] = (best_move, best_val)
		return self.memo[game_state]


class MinimaxAgentTEST1:
	def __init__(self):
		self.memo = {} #state, value
		#returns value, action

	def minimax(self, game_state):
		if game_state not in self.memo: #already visited this state?

			result = game_state.check_result()
			if result is not None: #leaf node or end of search
				best_move = None
				best_val = result * game_state.acting_player #return 0 for tie or 1 for maximizing win or -1 for minimizing win
				print('acting player endstate', game_state.acting_player)
				print('endstate val', best_val)

			else:
				print('game state')
				print(game_state)
				best_val = float('-inf')
				for i in game_state.available_moves():
					clone_state = copy.deepcopy(game_state)
					clone_state.make_move(i) #makes move and switches to next player
					print('clone state')
					print(clone_state)
					_, val = self.minimax(clone_state)
					val *= -1 #$clone_state.acting_player
					print('best val', best_val)
					print('val', val)
					print('action', i)		
					print('player', clone_state.acting_player)
					if val > best_val:
						best_move = i
						best_val = val	

			self.memo[game_state] = (best_move, best_val)
		return self.memo[game_state]

class MinimaxAgentTEST:
	def __init__(self):
		self.memo = {} #state, value
		#returns value, action

	def minimax(self, game_state):
		winning_moves = {}
		drawing_moves = {}
		losing_moves = {}
		if game_state not in self.memo: #already visited this state?

			result = game_state.check_result()
			if result is not None: #leaf node or end of search
				best_move = None
				best_val = result * game_state.acting_player #return 0 for tie or 1 for maximizing win or -1 for minimizing win
				print('acting player', game_state.acting_player)
				print(best_val)

			else:
				#best_val = float('-inf')
				for i in game_state.available_moves():
					clone_state = copy.deepcopy(game_state)
					clone_state.make_move(i) #makes move and switches to next player
					print(clone_state)
					_, val = self.minimax(clone_state)
					val *= -1 #$clone_state.acting_player
					print(val)
					if val == 1:
						winning_moves[i] = val
					elif val == 0:
						drawing_moves[i] = val
					elif val == -1:
						losing_moves[i] = val					
					# if val > best_val:
					# 	best_move = i
					# 	best_val = val

			if winning_moves:
				print('win', winning_moves)
				best_move = random.choice(list(winning_moves))
				best_val = winning_moves[best_move]
			elif drawing_moves:
				print('draw', drawing_moves)
				best_move = random.choice(list(drawing_moves))
				best_val = drawing_moves[best_move]
			elif losing_moves:
				print('lose', losing_moves)
				best_move = random.choice(list(losing_moves))
				best_val = losing_moves[best_move]
			self.memo[game_state] = (best_move, best_val)
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
	#ttt = Tictactoe([0,0,-1,0,0,0,1,-1,1])
	ttt = Tictactoe()
	#ttt = Tictactoe([1,0,0,-1,0,0,1,0,-1])
	#ttt = Tictactoe([1,0,0,0,0,0,0,0,0])
	#ttt = Tictactoe([1,-1,1,1,-1,-1,0,1,0],-1)
	#ttt = Tictactoe([-1,1,-1,-1,1,1,0,-1,0])
	#ttt = Tictactoe([0,0,-1,1,0,0,0,0,0],-1)
	#ttt = Tictactoe([1,0,0,0,0,0,0,0,-1])
	#print(ttt)
	mms = MinimaxAgent()
	h = HumanAgent()
	#print(mms.minimax(ttt))
	#agent = MinimaxAgent()
	#print(agent.select_move(ttt, 1))
	moves = 0
	while ttt.available_moves():
		print(ttt)
		print('\n')
		print('move', moves)
		print('acting player', ttt.acting_player)
		if moves % 2 == 0:
			print(ttt.board)
			move, _ = mms.minimax(ttt)
			print('minimax move', move)
		else:
			move = h.select_move(ttt)
		ttt.make_move(move)
		if ttt.check_result() == 0:
			print('Draw game')
			break
		elif ttt.check_result() == 1:
			print('Player 1 wins')
		elif ttt.check_result() == -1:
			print('Player 2 wins')
		moves+=1





