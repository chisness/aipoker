import random
from kuhn3p import betting, deck, Player

class Custom(Player):
        
	def __init__(self, rng=random.Random()):
		self.rng   = rng
		self.betting_round = 0

                # A list of where each player is seated for the round
		self.agent_p1_p2_positions = []

                # Checks track of the total number of a certain action each player performed
		self.raises = [0,0,0]
		self.calls = [0,0,0]
		self.checks = [0,0,0]
		self.folds = [0,0,0]

		# The likelyhood a raise is a bluff (must be divided by self.raise)
		self.raise_bluff = [0,0,0]

                # Creates  2-D arrays; an array for each acting state
                # this stores what the result of a match was given the agent's choices
                self.state_results = self.two_d_array(betting.num_internal())

                # A group of 2-D arrays saying what state occured in the round and its outcome
                self.changed_states = []

        def __str__(self):
		return 'Learning Agent'

        def two_d_array(self, size):
                array = []
                for x in range(size):
                        array.append([0,0])
                return array

	def start_hand(self, position, card):
                self.betting_round += 1
                self.agent_p1_p2_positions = [position, (position + 1) % 3, 
                                        (position + 2) % 3]

                self.changed_states = []

	def end_hand(self, position, card, state, shown_cards):
                shown_cards = self.reorder(shown_cards)
                actions = self.reorder(self.get_actions(state))
                winner = self.winner(actions, shown_cards)
                self.adjust(state, shown_cards, actions, winner)

        # Always bet on Ace. Always fold on Jack
	def act(self, state, card):
                action = 0
                num_players = 0
		if betting.can_bet(state):
                        if self.rng.random() >= self.algo(card, state):
                                action = betting.BET
                        else:
                                action = betting.CALL

		else:
                        if self.rng.random() >= 1.1 * self.algo(card, state):
                                action = betting.CALL
                        else:
                                action = betting.FOLD

                self.changed_states.append([state, action])
                
                return action

        # Returns the likelyhood of performing action 1 (either a check or a fold)
        def algo(self, card, state):
                chance = self.card_odds(card, state)
                return chance

        # The model for the learning algorithm
        def state_percent_model(self, state_result):
                if state_result[0] == state_result[1]:
                        return 0.5
                else:
                        m = max(state_result)
                        n = min(state_result)
                        s = m - n
                        if state_result[0] > state_result[1]:
                                return (s+1)/((s+1.0)**2)
                        else:
                                return 1.0 - (s+1)/((s+1.0)**2)

        # This is where the agent updates its knowledge base after each round
        def adjust(self, state, shown_hands, actions, winner):
                payoff = 0
                if winner == 0:         # if the agent wins
                        payoff = -1 * betting.pot_contribution(state, self.agent_p1_p2_positions[0])
                else:
                        payoff = betting.pot_size(state)

                # record the results for those states 
                for x in range(len(self.changed_states)):
                        state_id = self.changed_states[x][0]
                        action_id = self.changed_states[x][1]
                        self.state_results[state_id][action_id] += payoff

                for x in range(len(self.agent_p1_p2_positions)):
                        if actions[x].find("r") != -1:
                                self.raises[x] += 1
                                if shown_hands[x] != None:
                                        self.raise_bluff[x] += self.card_odds(shown_hands[x], None, shown_hands)
                        if actions[x].find("k") != -1:
                                self.checks[x] += 1
                        if actions[x].find("c") != -1:
                                self.calls[x] += 1
                        if actions[x].find("f") != -1:
                                self.folds[x] += 1
                

        # Determines the odds of anyone having a higher card than the agent
        def card_odds(self, card, state=None, shown_hands=None):
                assert state  != None or shown_hands != None

                odds = (deck.ACE - card) / 3.0
                if state != None:
                        odds *= self.other_players(state)
                else:
                        odds *= (3.0 - self.num_folds(shown_hands))
                        
                if odds >= 1.0:
                        return 1.0
                else:
                        return odds

        # Returns the number of other players that havent folded
        def other_players(self, state):
                if betting.facing_bet_fold(state):
                        return 1
                else:
                        return 2

        # Determines the number of folds in a given hand
        def num_folds(self, shown_hands):
                sum_folds = 0
                for x in range(len(shown_hands)):
                        if shown_hands[x] == None:
                                sum_folds += 1
                return sum_folds

        # Determines the winner of this round
        def winner(self, actions, shown_cards):                
                # If everyone folded, determine winner by who raised
                if shown_cards == [None, None, None]:
                        return actions.index("r")
                else:
                        return shown_cards.index(max(shown_cards))

        # Reorders game arrays so that the agent is in position 0
        def reorder(self, object_list):
                new_list = []
                for x in range(len(self.agent_p1_p2_positions)):
                        new_list.append(object_list[self.agent_p1_p2_positions[x]])
                return new_list

        # Returns what actions each agent commited as an array
        def get_actions(self, state):
                old_actions = betting.to_string(state)
                actions = ""

                is_check = True
                for x in range(len(old_actions)):
                        if old_actions[x] == "r":
                                is_check = False

                        if is_check is True and old_actions[x] == "c":
                                actions += "k"
                        else:
                                actions += old_actions[x]
                
                new_list = ["", "", ""]
                for x in range(len(actions)):
                        new_list[x%3] += actions[x]
                return new_list

