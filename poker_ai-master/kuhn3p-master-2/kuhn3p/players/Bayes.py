#!/bin/python
import random
from kuhn3p import betting, deck, Player

class Bayes(Player):
    def __init__(self, rng=random.random()):
        self.rng = rng
        self.uncertainty = [0, 0]

    def act(self, state, card):
        self.win_chance = self.win_heuristic(state, card)
        if betting.can_bet(state):
            return betting.BET if self.rng < self.win_chance else betting.CHECK
        else:
            return betting.CALL if self.rng < self.win_chance else betting.FOLD

    def end_hand(self, position, card, state, shown_cards):
        if state == betting.num_internal(): return
        if betting.bettor(state) == position: return
        if not betting.is_showdown(state): return
        assert shown_cards[betting.bettor(state)] is not None
        bluff = 0
        # If the bet was raised and it was not an Ace, remember the bluff
        if shown_cards[betting.bettor(state)] != deck.ACE:
            bluff = 1
        opp_pos = self.opponent_position(position, betting.bettor(state))
        self.uncertainty[opp_pos] = (9.0 * self.uncertainty[opp_pos] + bluff) / 10.0

    def chance_to_win(self, card, uncertain_cards):
        if card == deck.ACE: return 1.0
        elif card == deck.KING: return 1.0 / uncertain_cards
        else: return 0.0

    def opponent_position(self, position, opponent):
        if position == 0:
            opp_pos = 0 if opponent == 1 else 1
        elif position == 1:
            opp_pos = 0 if opponent == 2 else 1
        else:
            opp_pos = opponent
        return opp_pos

    def win_heuristic(self, state, card):
        chance_to_win = 0.0
        uncertainty_factor = 1.0
        if card == deck.ACE:
            chance_to_win = 1.0  #Ace will pay out
        elif betting.can_bet(state):
            chance_to_win = self.chance_to_win(card, 3)
        elif betting.can_fold(state) and not betting.call_closes_action(state):
            chance_to_win = self.chance_to_win(card, 2)
            uncertainty_factor = self.uncertainty[1]
        elif betting.facing_bet_call(state):
            chance_to_win = self.chance_to_win(card, 1)
            uncertainty_factor = self.uncertainty[0] * self.uncertainty[1]
        elif betting.facing_bet_fold(state):
            chance_to_win = self.chance_to_win(card, 1)
            uncertainty_factor = self.uncertainty[0]
        return chance_to_win * uncertainty_factor

    def __str__(self):
		return 'Bayes(rng=%f)' % (self.rng)