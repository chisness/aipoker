# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:48:52 2020

@author: mthom
"""


import numpy as np
import pandas as pd
from poker import hand_strength, rank, suit, hole_ranks

pd.options.display.max_columns = 50

def generate_hands(n, gen_q_matrix):

    def random_choice_noreplace(m,n, axis=-1):
        # m, n are the number of rows, cols of output
        return np.random.rand(m,n).argsort(axis=axis)
    
    #Deal Hands
    hands = random_choice_noreplace(n,52)
    hands = hands[:,:7]
    
    ranks = rank(hands)
    suits = suit(hands)
    
    player_flop = [hand_strength(ranks[i], suits[i], True) for i in range(len(ranks))]
    
    player_flop = pd.DataFrame(player_flop, columns=['p1_hand_value', 'p1_sub_value', 'p2_hand_value', 'p2_sub_value','result'])
    hole = hole_ranks(ranks.copy(), suits.copy())
    df = pd.concat([hole, player_flop], axis=1)
    df['ranks'] = ranks
    df['suits'] = suits    
    
    #7,8 decision only = 8 (2x4) 
    #6 full house type = 546 (6 * 91) 
    #4,5 = 1820 (2*91 * 10)
    #3 = 1092 (91 * 10)
    #2 = 1820: 91 * 11
    #1 = 1092: 91 * 19 
    #0 = 910 = 91 * 10
    
    #27,672 total states: 4*(2+91*(6+2*10+10+11+19+10)) 
    #aim for 14M simulations = ~500 per state (assuming equal probability)
    
    #player 0,1
    #hand_value
    #unsuited_hole_rank
    #hand sub_value
    #action : check/call/bet/reraise (or call_reraise)
    flop = np.zeros((2,4,9,91,20,4)) 
    pre = np.zeros((2, 169, 4))
    q = [pre, flop]

    if gen_q_matrix:
        return df, q
    else:
        return df


