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
			self.acting_player = 3 - self.acting_player #players are 2 or 1
			
	def available_moves(self):
		return [i for i in range(9) if self.board[i] == 0]
	
	def check_result(self):
		for (a,b,c) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
			if self.board[a] == self.board[b] == self.board[c] != 0:
				return 1
		if self.available_moves == []: return 0.5 #Tie
		return None #Game not over
		
	def __repr__(self):
		s= ""
		for i in range(9): 
			if self.board[i] == 0:
				s+=str(i+1)
			elif self.board[i] == 1:
				s+='x'
			elif self.board[i] == 2:
				s+='o'
			if i == 2 or i == 5: s += "\n"
		return s

class MinimaxAgent:
	def __init__(self)
		pass

	def select_move(self, game_state, depth = 9):
		winning_moves = []
		draw_moves = []
		losing_moves = []

		if depth == 0:
			return 

		for possible_move in game_state.legal_moves():
			next_state = game_state.apply_move(possible_move)
			if next_state.check_result():
				return -select_move(next_state, depth-1)
				
		if winning_moves:
			return random.choice(winning_moves)
		if draw_moves:
			return random.choice(draw_moves)
		return random.choice(losing_moves)


class HumanAgent:
	def select_move(self, game_state):
		print('Enter your move (1-9): ')
		move = int(float(input()))-1
		#print('move', move)
		#print('game state available moves', game_state.available_moves())
		if move in game_state.available_moves():
			return move
		else:
			print('Invalid move, try again')
			self.select_move(game_state)
	
class BasicAgent:
	def select_move(self, game_state):
		return random.choice(game_state.available_moves())

if __name__ == "__main__":
	ttt = Tictactoe()



