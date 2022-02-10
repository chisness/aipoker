import random
import matplotlib.pyplot as plt
import math
import sys, os
import numpy
from kuhn3p import betting, deck, Player
from time import sleep
import itertools
from .player_utilities import UTILITY_DICT
import pandas as pd
import json

plt.rcParams['agg.path.chunksize'] = 10000

handperm = list(itertools.permutations(
    [deck.JACK, deck.QUEEN, deck.KING, deck.ACE], 3))
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


tree = {
    'i': {
        'player': 0,
        'strategySum': {
            'k': 0,
            'b': 0
        },
        'regretSum': {
            'k': 0,
            'b': 0
        },
        'strategy': {
            'k': 0.5,
            'b': 0.5
        }
    },
    'ik': {
        'player': 1,
        'strategySum': {
            'k': 0,
            'b': 0
        },
        'regretSum': {
            'k': 0,
            'b': 0
        },
        'strategy': {
            'k': 0.5,
            'b': 0.5
        }
    },
    'ikk': {
        'player': 2,
        'strategySum': {
            'k': 0,
            'b': 0
        },
        'regretSum': {
            'k': 0,
            'b': 0
        },
        'strategy': {
            'k': 0.5,
            'b': 0.5
        },
    },
    'ikkb': {
        'player': 0,
        'strategySum': {
            'f': 0,
            'c': 0
        },
        'regretSum': {
            'f': 0,
            'c': 0
        },
        'strategy': {
            'c': 0.5,
            'f': 0.5
        }
    },
    'ikkbc': {
        'player': 1,
        'strategySum': {
            'f': 0,
            'c': 0
        },
        'regretSum': {
            'f': 0,
            'c': 0
        },
        'strategy': {
            'c': 0.5,
            'f': 0.5
        }
    },
    'ikkbf': {
        'player': 1,
        'strategySum': {
            'f': 0,
            'c': 0
        },
        'regretSum': {
            'f': 0,
            'c': 0
        },
        'strategy': {
            'c': 0.5,
            'f': 0.5
        }
    },
    'ikb': {
        'player': 2,
        'strategySum': {
            'f': 0,
            'c': 0
        },
        'regretSum': {
            'f': 0,
            'c': 0
        },
        'strategy': {
            'c': 0.5,
            'f': 0.5
        }
    },
    'ikbf': {
        'player': 0,
        'strategySum': {
            'f': 0,
            'c': 0
        },
        'regretSum': {
            'f': 0,
            'c': 0
        },
        'strategy': {
            'c': 0.5,
            'f': 0.5
        }
    },
    'ikbc': {
        'player': 0,
        'strategySum': {
            'f': 0,
            'c': 0
        },
        'regretSum': {
            'f': 0,
            'c': 0
        },
        'strategy': {
            'c': 0.5,
            'f': 0.5
        }
    },
    'ib': {
        'player': 1,
        'strategySum': {
            'f': 0,
            'c': 0
        },
        'regretSum': {
            'f': 0,
            'c': 0
        },
        'strategy': {
            'c': 0.5,
            'f': 0.5
        }
    },
    'ibf': {
        'player': 2,
        'strategySum': {
            'f': 0,
            'c': 0
        },
        'regretSum': {
            'f': 0,
            'c': 0
        },
        'strategy': {
            'c': 0.5,
            'f': 0.5
        }
    },
    'ibc': {
        'player': 2,
        'strategySum': {
            'f': 0,
            'c': 0
        },
        'regretSum': {
            'f': 0,
            'c': 0
        },
        'strategy': {
            'c': 0.5,
            'f': 0.5
        }
    }
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

multiple = 2        

dir_path = os.path.dirname(os.path.realpath(__file__))


class UltimateAiKhun(Player):
    def __init__(self):
        self.player = -1
        self.card = -1
        self.tables = tree
        self.avg_strategy = Strategy
        self.score_perf = []
        self.player_strategy = {}
        self.card_profile = {}
        self.p = 0
        if len(sys.argv) > 1 and sys.argv[1] == 'train':
            self.train_cfr()

    def train_cfr(self):
        iterations = int(sys.argv[2])
        stats_range = int(iterations / multiple) + 1
        s = 0
        t = 0
        p1 = 1
        p2 = 1
        p3 = 1
        hits = 0

        print(stats_range)

        self.performance = {
            'i': [0 for _ in range(stats_range)],
            'ik': [0 for _ in range(stats_range)],
            'ikk': [0 for _ in range(stats_range)],
            'ikkb': [0 for _ in range(stats_range)],
            'ikkbc': [0 for _ in range(stats_range)],
            'ikkbf': [0 for _ in range(stats_range)],
            'ikb': [0 for _ in range(stats_range)],
            'ikbf': [0 for _ in range(stats_range)],
            'ikbc': [0 for _ in range(stats_range)],
            'ib': [0 for _ in range(stats_range)],
            'ibf': [0 for _ in range(stats_range)],
            'ibc': [0 for _ in range(stats_range)],
        }

        score = []
        while t < iterations:
            i = 2
            self.training_hand = random.choice(handperm)
            while i >= 0:
                score.append(self.cfr('i', i, t, p1, p2, p3))
                i -= 1
            print(t)
            t += 1
            if t % multiple == 0:            
                self.p += 1                
            

        self.get_average_strategy()
        data = pd.Series(self.avg_strategy)
        self.tables = tre
        print(self.avg_strategy)
        data.to_csv(dir_path + '/strategies/substrategy' +
                    str(p1) + '|' + str(p2) + '|' + str(p3) + str(s) + '.csv')
        
        plt.plot(self.performance['i'])
        
        plt.show()

    def start_hand(self, position, card):
        global game_t
        game_t += 1
        self.player = position
        self.card = card
        print('ultimate card', self.card)
        player_key = 'strategy' + str(self.player)
        if player_key in self.player_strategy:
            self.avg_strategy = self.player_strategy[player_key]
        else:
            data = pd.read_csv(dir_path + '/strategies/strategy' +
                               # str(random.randint(0, 0)) +
                               '.csv', header=None, index_col=0, delimiter=',')
            json_dict = data[1].to_dict()

            for k in json_dict:
                self.avg_strategy[k] = json.loads(json_dict[k].replace(
                    "'", '#').replace('"', "'").replace('#', '"'))

            # self.player_strategy[player_key] = self.avg_strategy

    def extract(self, x):
        return True

    def act(self, state, card, node=None):
        global game_t
        decision = -1
        search_nodes = []
        profile_key = ''
        if node is not None:
            key = state_map[node] if node else state_map['i']
            node_weights = self.avg_strategy[key]
            node_strategy = [node_weights[k] for k in node_weights]
            if card == deck.ACE:
                if betting.can_bet(state):
                    return numpy.random.choice([0, 1], p=[.1, .9])
                elif betting.facing_bet(state):
                    return 0
            if card == deck.JACK:
                if betting.can_bet(state):
                    return numpy.random.choice([0, 1], p=[.9, .1])
                elif betting.facing_bet(state):
                    return numpy.random.choice([1, 0], p=[.9, .1])

            search_key = str(self.player) + str(self.card) + key
            decision = numpy.random.choice([0, 1], p=node_strategy)

            # if betting.facing_bet(state) and game_t > 300 and key in self.card_profile:
            #     length = len(key)
            #     if key[length - 1] != 'f':
            #         search_nodes.append(key[:length - 1])
            #     if key[length - 1] == 'c':
            #         search_nodes.append(key[:length - 2])
            #     profile_key = k

            #     cards_dist = {}
            #     cards_pred = {}

            #     if search_nodes:
            #         card_profile = self.card_profile
            #         for k in search_nodes:
            #             if k in card_profile:
            #                 cards_dist[k] = card_profile[k]
            #                 cards_pred[k] = max(cards_dist[k])
            #                 # print(key, cards_dist)

            #         for k in cards_dist:
            #             normalization = float(sum(cards_dist[k]))
            #             freq = float(cards_pred[k])
            #             prob = numpy.divide(freq, normalization)
            #             cards_pred[k] = cards_dist[k].index(cards_pred[k])
            #             cards_dist[k] = prob

            #         bet_card = cards_pred[key[:length - 1]]
            #         prob = cards_dist[key[:length - 1]]
            #         if card > bet_card:
            #             return numpy.random.choice([0, 1], p=[prob, 1 - prob])

        return decision

    def end_hand(self, position, card, state, shown_cards):
        play_string = betting.to_string(state)
        h = state_map[play_string]
        profile_key = h
        node = ''
        for i in h[:len(h) - 1]:
            node += i
            card = shown_cards[position]
            p = self.tables[node]['player']
            if p != position:
                c = shown_cards[p]
                if c is not None:
                    if node in self.card_profile:
                        self.card_profile[node][c] += 1
                    else:
                        self.card_profile[node] = [0, 0, 0, 0]
                        self.card_profile[node][c] += 1

    def cfr(self, h, i, t, pi, pi2, pi3):
        """
        @Description: counter factual regret recursive function
        @Params:
            h: action history
            i: player i
            t: iteration
            pi: probability profile for player i
            pni: probability profile without player i
        """
        if h not in Actions:
            return self.utility(h, i, self.training_hand)

        vsigma = 0
        Vsigma = {
            'c': 0,
            'b': 0,
            'k': 0,
            'f': 0
        }

        if i == 0:
            self.update_table(h, pi)
        elif i == 1:
            self.update_table(h, pi2)
        elif i == 2:
            self.update_table(h, pi3)

        for a in Actions[h]:
            if i == 0:
                Vsigma[a] = self.cfr(
                    h + a, i, t, self.tables[h]['strategy'][a] * pi, pi2, pi3)
            elif i == 1:
                Vsigma[a] = self.cfr(
                    h + a, i, t, pi, pi2 * self.tables[h]['strategy'][a], pi3)
            elif i == 2:
                Vsigma[a] = self.cfr(
                    h + a, i, t, pi, pi2, self.tables[h]['strategy'][a] * pi3)

            vsigma += self.tables[h]['strategy'][a] * Vsigma[a]

        for a in Actions[h]:
            if i == 0:
                regret = pi2 * pi3 * (Vsigma[a] - vsigma)
            elif i == 1:
                regret = pi * pi3 * (Vsigma[a] - vsigma)
            elif i == 2:
                regret = pi * pi2 * (Vsigma[a] - vsigma)

            self.performance[h][self.p] = regret

            assert not math.isnan(regret)
            self.tables[h]['regretSum'][a] += regret

        return vsigma

    def update_table(self, h, pi):
        normalization = 0
        actionregret = {
            'c': 0,
            'b': 0,
            'k': 0,
            'f': 0
        }
        for a in Actions[h]:
            actionregret[a] = max([self.tables[h]['regretSum'][a], 0])
            normalization += actionregret[a]

        for a in Actions[h]:
            self.tables[h]['strategy'][a] = actionregret[a] / \
                normalization if normalization > 0 else 0.5
            self.tables[h]['strategySum'][a] += pi * \
                self.tables[h]['strategy'][a]

    def get_average_strategy(self):
        for h in Actions:
            normalization = 0
            for a in Actions[h]:
                normalization += self.tables[h]['strategySum'][a]

            for a in Actions[h]:
                self.avg_strategy[h][a] = self.tables[h]['strategySum'][a] / \
                    normalization if normalization > 0 else 0.5

    def utility(self, h, i, hand):
        return UTILITY_DICT.get(h[1:])(i, hand)

    def __str__(self):
        return 'UltimateAiKhun'
