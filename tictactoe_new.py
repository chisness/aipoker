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
	
class MMNode:
	def __init__(self, game_state):
		self.game_state = game_state
		self.children = []
		self.acting_player = game_state.acting_player
		self.add_children(game_state)
		
	def add_children(self, game_state):
		if game_state.check_result() > 0:
			return
		move_index = game_state.available_moves()
		for i in move_index:
			new_state = game_state.make_move(move_index[i])
			self.children.append(MMNode(Tictactoe(board = new_state)))
			
	def minimax(self, depth = 9)
		if len(self.children) == 0 or depth == 0:
			return self.game_state.check_result()
		child_states = max([-minimax(child, depth - 1) for child in self.children]
		
	def nh_val(self):
		return sum([self.game_state.check_result() for child in self.children]) / len(self.children)
		
		
class Node: 
	def __init__(self, game_state, parent = None, move = None):
		self.parent = parent
		self.move = move
		self.game_state = game_state
		self.children = []
		self.win_count = 0
		self.num_rollouts = 0
		print('new node')
		print(type(game_state))
		self.unvisited_moves = game_state.available_moves()
		self.player = game_state.acting_player
		
	def add_random_child(self, game_state):
		move_index = random.randint(0, len(self.unvisited_moves)-1)
		print('move index', move_index)
		new_move = self.unvisited_moves.pop(move_index)
		print('new mnove', new_move)
		new_game_state = self.game_state.make_move(new_move)
		print('stone', type(new_game_state))
		new_node = Node(new_game_state, self, new_move)
		print('test')
		print(type(new_node))
		self.children.append(new_node)
		return new_node
	

	
	def update(self, result):
		self.win_count += result
		self.num_rollouts += 1
		
class MinimaxAgent:
	def __init__(self)
		pass

	def select_move(self, game_state, depth = 9):
		winning_moves = []
		draw_moves = []
		losing_moves = []

		for possible_move in game_state.legal_moves():
			next_state = game_state.apply_move(possible_move)
			if next_state.check_result():
				


		if depth == 0 or len(node.children) == 0:
			return (node.game_state, node.nh_val)
		return (node.game_state, )
	
class UCTAgent:
	def __init__(self, num_rounds = 200, temperature = 1, alphabeta = False):
		self.num_rounds = num_rounds
		self.temperature = temperature
		self.alphabeta = alphabeta
		
	def select_move(self, game_state):
		print ('select move', game_state.acting_player)
		
		print(type(game_state))
		root = MCTSNode(game_state)
		
		for i in range(self.num_rounds):
			node = root
			#gs = copy.copy(game_state) #deepcopy?
			#print('gs')
			#print(type(gs))
			print('node test', node)
			
			
			#selection -- select leaf that has not had playout (simulation) yet
			#while (not node.can_add_child()) and (not node.is_terminal()):
			while node.unvisited_moves == [] and node.children != []:
				print('first')
				node = self.uct_select_child(node)
				print('node test2', node)
				#gs.make_move(node.move)
			
			#expansion -- expand from leaf unless leaf is end of game
			if node.unvisited_moves != []:
				print('second')
				node = node.add_random_child()
				print('node fucked up', node)
				print('fucked', type(node))
			
			#simulation -- complete a random playout from the newly expanded node
			while gs.available_moves() != []:
				print('third')
				gs.make_move(random.choice(gs.available_moves()))
			
			#backpropagation -- update all nodes from the selection to leaf stage
			while node is not None:
				node.update(gs.check_result(node.acting_player))
				node = node.parent
				
	def uct_select_child(self, node):
		total_rollouts = sum(child.num_rollouts for child in node.children)
		log_rollouts = math.log(total_rollouts)
		
		best_score = -1
		best_child = None
		for child in node.children:
			win_percentage = float(child.win_count)/float(child.num_rollouts)
			expl_factor = math.sqrt(log_rollouts / child.num_rollouts)
			uct_score = win_percentage + self.temperature * expl_factor
			if uct_score > best_score:
				best_score = uct_score
				best_child = child
		return best_child
		
class RandomAgent:	   
	def select_move(self, game_state):
		return random.choice(game_state.available_moves())
	
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
		#win if possible, block loss if possible, do double win option for next time if possible, otherwise random

if __name__ == "__main__":
	ttt = Tictactoe()
	#print(type(game_state))
	#game_state = Tictactoe(board = ['x','o',0,0,'x',0,'o',0,0])
	
	#agent1 = HumanAgent(num_rounds = 1000, temperature = 1, alphabeta = False)
	#agent2 = HumanAgent(num_rounds = 1000, temperature = 1, alphabeta = False)
	agent1 = HumanAgent()
	agent2 = HumanAgent()

	#ttt.play(agent1, agent2)
	
	
	while (ttt.available_moves() != []):
		print('Player: {} (1 is x, 2 is o)'.format(ttt.acting_player))
		print(ttt)
		if ttt.acting_player == 1:
			move = agent1.select_move(ttt)
		else:
			move = agent2.select_move(ttt)
		ttt.make_move(move)
		result = ttt.check_result(ttt.acting_player)
		if result == 1.0:
			print('Player {} wins! '.format(ttt.acting_player))
			break
			#ttt = Tictactoe()
			#ttt.play(ttt, player1, player2)
		elif result == 0.5:
			print('Tie game! Starting next game now.')
			break
			#ttt = Tictactoe()
			#ttt.play(ttt, player1, player2)
		ttt.acting_player = 3 - ttt.acting_player
	