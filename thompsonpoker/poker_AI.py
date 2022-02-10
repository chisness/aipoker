# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:45:13 2020

@author: mthom
"""

import numpy as np


def preflop_decision(q, random_rate, player, bet):
    '''
    bet is 'none', 'bet', 'reraise'
    '''
    d0 = 'none'
    if np.random.uniform(0,1,1)>=random_rate:
        random_action_used = False
        if player == 1:
            if bet == 'none':
                if q[[0,2]].max()>-1:
                    d0 = ['check','bet'][q[[0,2]].argmax()]
                else:
                    d0 = 'fold'
            elif bet == 'bet':
                if  q[1]>=0:
                    d0 = 'call'
                else:
                    d0 = 'fold'
            elif bet == 'reraise':
                if q[3]>=0:
                    d0 = 'call_reraise'
                else:
                    d0 = 'fold'
        else:
            if bet == 'none':
                d0 = ['check','bet'][q[[0,2]].argmax()]
            elif bet == 'bet':
                if q[[1,3]].max()>=0:
                    d0 = ['call','reraise'][q[[1,3]].argmax()]
                else:
                    d0 = 'fold'
    else:
        random_action_used = True
        if bet == 'none':
            if player == 0:
                d0 = np.random.choice(['check', 'bet'])
            else:
                d0 = np.random.choice(['fold', 'check', 'bet'])
        elif bet == 'bet':
            if player == 1:
                d0 = np.random.choice(['fold','call'])
            else:
                d0 = np.random.choice(['fold','call', 'reraise'])
        else:
            d0 = np.random.choice(['fold','call_reraise'])
    if d0 == 'none':
        raise Exception(q, random_rate, player, bet, random_action_used)
    return d0

    
def flop_decision(q, random_rate, player, bet):
    if np.random.uniform(0,1,1)>=random_rate:
        if player == 0:
            if bet == 'none':
                return ['check','bet'][q[[0,2]].argmax()]
            elif bet == 'bet':
                if q[1]>=0:
                    return 'call'
                else:
                    return 'fold'
            elif bet == 'reraise':
                if q[3]>=0:
                    return 'call_reraise'
                else:
                    return 'fold'
        elif player == 1:
            if bet == 'none':
                return ['check','bet'][q[[0,2]].argmax()]
            elif bet == 'bet':
                if q[[1,3]].max()>=0:
                    return ['call','reraise'][q[[1,3]].argmax()]
                else:
                    return 'fold'
    else:     
        if player == 0:
            if bet == 'none':
                return np.random.choice(['check','bet'])
            elif bet == 'bet':
                return np.random.choice(['fold','call'])
            elif bet == 'reraise':
                return np.random.choice(['fold','call_reraise'])
        elif player == 1:
            if bet == 'none':
                return np.random.choice(['check','bet'])
            elif bet == 'bet':
                return np.random.choice(['fold','call','reraise'])
#######################   END #######################


#######################   Player is in First Position Formulas #######################
def first_position_ai_preflop_decisions(q, df, player_chips, ai_chips, pot, bet_type, random_rate=0):
    q_values = q[0][1, df['rank2'], :]
    action = preflop_decision(q_values, random_rate, 1, bet_type)
    
    if action == 'fold':
        player_chips += pot
    elif action == 'check':
        pot += 1
        ai_chips += -1
    elif action == 'bet':
        pot += 3
        ai_chips += -3
    elif action == 'call':
        pot += 2
        ai_chips += -2
    elif action == 'call_reraise':
        pot += 4
        ai_chips += -4
    else:
        raise Exception('Unexpected AI Action')
    return player_chips, ai_chips, pot, action
    
      

def first_position_ai_flop_decision(q, pre_flop_bet, df, player_chips, ai_chips, pot, bet_type, random_rate=0):
    q_values = q[1][1, pre_flop_bet, df['p2_hand_value'], df['suitless2'], df['p2_sub_value'], :]
    action = flop_decision(q_values, 0, 1, bet_type)
    
    if action == 'fold':
        player_chips += pot
    elif action == 'check':
        pass
    elif action == 'bet':
        pot += 4
        ai_chips += -4
    elif action == 'call':
        pot += 4
        ai_chips += -4
    elif action == 'reraise':
        pot += 12
        ai_chips += -12
    else:
        raise Exception('Unexpected AI Action')
    return player_chips, ai_chips, pot, action



def first_position_show_down(df, pot, ai_chips, player_chips, player_call):
    hand_dict = {0:'High Card', 1:'Pair', 2:'Two Pair', 3:'Three of a kind', 4:'Straight', 5:'Flush', 6:'Full House', 7:'Four of a kind',8:'Straight Flush'}
    
    show_ai_cards = False
    if player_call == True:
        show_ai_cards = True
    
    if df['result'] == 1:
        player_chips +=  pot
        if df['p1_hand_value'] in [1, 4, 5, 6, 8]:
            result_text = 'You win with a {}'.format(hand_dict[df['p1_hand_value']])
        else:
            result_text = 'You win with {}'.format(hand_dict[df['p1_hand_value']])
    
    elif df['result'] == -1:
        show_ai_cards = True
        ai_chips += pot
           
        if df['p2_hand_value'] in [1, 4, 5, 6, 8]:
            result_text = 'AI wins with a {}'.format(hand_dict[df['p2_hand_value']])
        else:
            result_text = 'AI wins with {}'.format(hand_dict[df['p2_hand_value']])
    
    elif df['result'] == 0:
        show_ai_cards = True
        ai_chips += int(pot/2)
        player_chips += int(pot/2)
        if df['p2_hand_value'] in [1, 4, 5, 6, 8]:
            result_text = 'Split pot, both players have a {}'.format(hand_dict[df['p2_hand_value']])
        else:
            result_text = 'Split pot, both players have {}'.format(hand_dict[df['p2_hand_value']])
    else:
        raise Exception()
    
    return player_chips, ai_chips, result_text, show_ai_cards
     
#######################   Player is the Dealer Formulas #######################            
def dealer_ai_preflop_decision(q, df, player_chips, ai_chips, pot, action, random_rate=0):
    if action == 'bet':
        bet_type = 'bet'
    else:
        bet_type = 'none'
        
    q_values = q[0][0, df['rank1'], :]
    action = preflop_decision(q_values, random_rate, 0, bet_type)
    
    if action == 'call' or action == 'bet':
        pot += 2
        ai_chips += -2
    elif action == 'reraise':
        ai_chips += -6
        pot += 6
    elif action == 'fold':
        player_chips += pot
        
    return player_chips, ai_chips, pot, action


def dealer_ai_flop_first_decision(q, pre_flop_bet, df, pot, ai_chips):
    #p2's respsone to p1's bet/reraise (their 2nd preflop decision, 3rd total preflop action)

    q_values = q[1][0, pre_flop_bet, df['p1_hand_value'], df['suitless1'], df['p1_sub_value'], :]
    action = flop_decision(q_values, 0, 0, 'none')
    
    if action == 'bet':
        ai_chips += -4
        pot += 4 
        
    return ai_chips, pot, action
            

def dealer_ai_flop_response(q, pre_flop_bet, df, pot, ai_chips, player_chips, bet_type):
    q_values = q[1][0, pre_flop_bet, df['p1_hand_value'], df['suitless1'], df['p1_sub_value'], :]
    action = flop_decision(q_values, 0, 0, bet_type)
    
    if action == 'call':
        ai_chips += -4
        pot += 4
    elif action == 'call_reraise':
        ai_chips += -8
        pot += 8
    elif action == 'fold':
        player_chips += pot
    else:
        raise Exception('Unexpected AI Action')
    
    return pot, ai_chips, player_chips, action            


def dealer_show_down(df, pot, ai_chips, player_chips, player_call):
    hand_dict = {0:'High Card', 1:'Pair', 2:'Two Pair', 3:'Three of a kind', 4:'Straight', 5:'Flush', 6:'Full House', 7:'Four of a kind',8:'Straight Flush'}
    
    show_ai_cards = False
    if player_call:
        show_ai_cards = True
        
    if df['result'] == 1:
        show_ai_cards = True
        ai_chips +=  pot
        if df['p1_hand_value'] in [1, 4, 5, 6, 8]:
            result_text = 'AI wins with a {}'.format(hand_dict[df['p1_hand_value']])
        else:
            result_text = 'AI wins with {}'.format(hand_dict[df['p1_hand_value']])
    
    
    elif df['result'] == -1 :
        player_chips += pot
           
        if df['p2_hand_value'] in [1, 4, 5, 6, 8]:
            result_text = 'You win with a {}'.format(hand_dict[df['p2_hand_value']])
        else:
            result_text = 'You win with {}'.format(hand_dict[df['p2_hand_value']])
    
    elif df['result'] == 0:
        show_ai_cards = True
        ai_chips += int(pot/2)
        player_chips += int(pot/2)
        if df['p2_hand_value'] in [1, 4, 5, 6, 8]:
            result_text = 'Split pot, both players have a {}'.format(hand_dict[df['p2_hand_value']])
        else:
            result_text = 'Split pot, both players have {}'.format(hand_dict[df['p2_hand_value']])
    else:
        raise Exception()
    
    return player_chips, ai_chips, result_text, show_ai_cards

