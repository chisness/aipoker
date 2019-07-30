class Player(object):
    def __init__(self):
        pass

    def start_hand(self, position, card):
        pass

    def act(self, state, card):
        raise NotImplementedError

    def end_hand(self, position, card, state, shown_cards):
        pass
