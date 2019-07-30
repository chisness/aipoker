import random
import matplotlib.pyplot as plt
import math
import sys
import numpy
from kuhn3p import betting, deck, Player
from time import sleep
import itertools
from .player_utilities import UTILITY_DICT
import pandas as pd
import json
from .weights import weights

plt.rcParams['agg.path.chunksize'] = 10000


Actions = {
    'i': ('k', 'b'),
    'ik': ('k', 'b'),
    'ikk': ('k', 'b'),
    'ikkb': ('c', 'f'),
    'ikkbc': ('c', 'f'),
    'ikkbf': ('c', 'f'),
    'ikb': ('c', 'f'),
    'ikbf': ('c', 'f'),
    'ikbc': ('c', 'f'),
    'ib': ('c', 'f'),
    'ibf': ('c', 'f'),
    'ibc': ('c', 'f'),
}



Strategy = dict()


state_map = {
    'i':       'i',
    'c':       'ik',
    'cc':      'ikk',
    'ccc':     'ikkk',
    'ccr':     'ikkb',
    'ccrf':    'ikkbf',
    'ccrc':    'ikkbc',
    'cr':      'ikb',
    'crf':     'ikbf',
    'crc':     'ikbc',
    'r':       'ib',
    'rf':      'ibf',
    'rc':      'ibc',
    'crcc':    'ikbcc',
    'ccrcf':  'ikkbcf',
    'ccrcc':   'ikkbcc',
    'ccrfc': 'ikkbfc',
    'ccrff': 'ikkbff',
    'crfc':    'ikbfc',
    'crff':    'ikbff',
    'crcf':   'ikbcf',
    'crcc':  'ikbcc',
    'rfc': 'ibfc',
    'rcc': 'ibcc',
    'rcf':   'ibcf',
    'rff':   'ibff',
}


action_map = {
    'k': 0,
    'c': 0,
    'b': 1,
    'f': 1
}

for a in Actions:
    Strategy[a] = dict([[k, 0] for k in Actions[a]])

game_t = 0

class UltimateAiKhun2(Player):
    def __init__(self, version):
        self.player = -1
        self.card = -1
        self.version = version


    def start_hand(self, position, card):
        global game_t
        game_t += 1
        self.player = position
        self.card = card

    def act(self, state, card, node=None):
        global game_t
        if node is not None:
            key = state_map[node] if node else state_map['i']
            card_key = str(card)
            strategy = weights[self.version][key]
            return numpy.random.choice([0, 1], p=strategy[card_key])
        return 0

    def end_hand(self, position, card, state, shown_cards):
        play_string = betting.to_string(state)
        h = state_map[play_string]
        # print state, shown_cards

    def __str__(self):
        return 'UltimateAiKhun'
