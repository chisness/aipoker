import numpy as np

def brf(self, player_card, history, player_iteration, opp_reach):
	plays = len(history)
	acting_player = plays % 3
	expected_payoff = 0

	if plays >= 3: #can be terminal
		opponent_dist = np.zeros(len(opp_reach))
		opponent_dist_total = 0
		#print('opp reach', opp_reach)
		if history[-1] == 'f' or history[-1] == 'c' or (history[-1] == history[-2] == 'k'):
			for i in range(len(opp_reach)):
				opponent_dist_total += opp_reach[i] #compute sum of dist. for normalizing
			for i in range(len(opp_reach)):
				opponent_dist[i] = opp_reach[i] / opponent_dist_total
				payoff = 0
				is_player_card_higher = player_card > i
				if history[-1] == 'f': #bet fold
					if acting_player == player_iteration:
						payoff = 1
					else:
						payoff = -1
				elif history[-1] == 'c': #bet call
					if is_player_card_higher:
						payoff = 2
					else:
						payoff = -2
				elif (history[-1] == history[-2] == 'k'): #check check
					if is_player_card_higher:
						payoff = 1
					else:
						payoff = -1
				expected_payoff += opponent_dist[i] * payoff
			return expected_payoff

	d = np.zeros(2) #opponent action distribution
	d = [0, 0]

	new_opp_reach = np.zeros(len(opp_reach))
	for i in range(len(opp_reach)):
		new_opp_reach[i] = opp_reach[i]

	v = -100000
	util = np.zeros(2)
	util = [0, 0]
	w = np.zeros(2)
	w = [0, 0]

	#infoset = history

	for a in range(2):
		if acting_player != player_iteration:
			for i in range(len(opp_reach)):
				infoset = str(i) + history 
				if infoset not in self.nodes:
					self.nodes[infoset] = Node(2)
				strategy = self.nodes[infoset].get_average_strategy()#get_strategy_br()
				new_opp_reach[i] = opp_reach[i] * strategy[a] #update reach prob
				w[a] += new_opp_reach[i] #sum weights over all poss. of new reach

		if a == 0:
			if len(history) != 0:
				if history[-1] == 'b':
					next_history = history + 'f'
				elif history[-1] == 'k':
					next_history = history + 'k'
			else:
				next_history = history + 'k'
		elif a == 1:
			if len(history) != 0:
				if history[-1] == 'b':
					next_history = history + 'c'
				elif history[-1] == 'k':
					next_history = history + 'b'
			else:
				next_history = history + 'b'
		#print('w', w)
		#print('history', history)
		#print('next history', next_history)
		util[a] = self.brf(player_card, next_history, player_iteration, new_opp_reach)
		#print('util a', util[a])
		if (acting_player == player_iteration and util[a] > v):
			v = util[a] #this action better than previously best action
	
	if acting_player != player_iteration:
		#D_(-i) = Normalize(w) , d is action distribution that = normalized w
		d[0] = w[0] / (w[0] + w[1])
		d[1] = w[1] / (w[0] + w[1])
		v = d[0] * util[0] + d[1] * util[1]

	return v