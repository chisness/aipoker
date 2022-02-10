import random

JACK = 0
QUEEN = 1
KING = 2
ACE = 3


def num_cards():
    return 4


def valid_card(card):
    return 0 <= card < num_cards()


__card_to_string = 'JQKA'


def card_to_string(card):
    assert valid_card(card)
    return __card_to_string[card]


def string_valid_card(string):
    return len(string) == 1 and string[0] in __card_to_string


def string_to_card(string):
    assert string_valid_card(string)
    return __card_to_string.find(string)


def shuffled(rng=random.Random()):
    cards = list(range(num_cards()))
    rng.shuffle(cards)
    return cards


if __name__ == "__main__":
    assert card_to_string(0) == 'J'
    assert card_to_string(1) == 'Q'
    assert card_to_string(2) == 'K'
    assert card_to_string(3) == 'A'

    assert string_to_card('J') == 0
    assert string_to_card('Q') == 1
    assert string_to_card('K') == 2
    assert string_to_card('A') == 3
