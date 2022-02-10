import random
from kuhn3p import betting, deck, Player


class Thief(Player):
    def __init__(self, bluff, rng=random.Random()):
        assert bluff >= 0 and bluff <= 1
        self.didBluff = False
        self.bluff = bluff
        self.rng = rng

    def start_hand(self, position, card):
        didBluff = False

    def act(self, state, card):
        if betting.can_bet(state):
            if card == deck.ACE:
                if betting.actor(state) == 0:
                    if self.rng.random() > self.bluff:
                        return betting.BET
                    else:
                        didBluff = True
                        return betting.CHECK
                elif betting.actor(state) == 1:
                    if self.rng.random() > self.bluff/2:
                        return betting.BET
                    else:
                        didBluff = True
                        return betting.CHECK
                else:
                    return betting.BET
            elif card == deck.KING:
                if self.rng.random() > self.bluff:
                    didBluff = True
                    return betting.BET
                else:
                    return betting.CHECK
            elif card == deck.QUEEN:
                if self.rng.random() < self.bluff:
                    didBluff = True
                    return betting.BET
                else:
                    return betting.CHECK
            else:
                if self.rng.random() < self.bluff/2:
                    didBluff = True
                    return betting.BET
                else:
                    return betting.CHECK
        else:
            if card == deck.ACE:
                return betting.CALL
            elif card == deck.KING:
                if self.bluff*1.5 > self.rng.random():
                    didBluff = True
                    return betting.CALL
                else:
                    return betting.FOLD
            else:
                return betting.FOLD

    def end_hand(self, position, card, state, shown_cards):
        best_card = -1
        for x in shown_cards:
            if(x > best_card):
                best_card = x
        if best_card == card and self.didBluff:
            #bluff was correct, increase bluff chance
            if self.bluff < .90:
                self.bluff += 0.15
        elif best_card != card and self.didBluff:
            #incorrect bluff, decrease bluff chance
            if self.bluff > 0.1:
                self.bluff -= 0.15


    def __str__(self):
        return 'Thief(bluff=%f)' % (self.bluff)
