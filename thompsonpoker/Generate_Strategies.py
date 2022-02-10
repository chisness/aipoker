# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:22:00 2020

@author: mthom
"""

import numpy as np
import pandas as pd
from Generate_Hands_and_States import generate_hands
import time

start = time.time()
print('generating hands...')

#actions: check/call/bet/[reraise, call_reraise]
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
                    if q[[0,2]].max() == 0: #start with aggression
                        d0 = 'bet'
                    else:
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
                if q[[0,2]].max()==0: #start with aggression
                    d0 = 'bet'
                else:
                    d0 = ['check','bet'][q[[0,2]].argmax()]
            elif bet == 'bet':
                if q[[1,3]].max()>=0:
                    if q[[1,3]].max()==0: #start with aggression
                        d0 = 'reraise'
                    else:
                        d0 = ['call','reraise'][q[[1,3]].argmax()]
                else:
                    d0 = 'fold'
    else:
        random_action_used = True
        if bet == 'none':
            d0 = np.random.choice(['check', 'bet'])
        elif bet == 'bet':
            if player == 1:
                d0 = 'call'
            else:
                d0 = np.random.choice(['call', 'reraise'])
        else:
            d0 = 'call_reraise'
    if d0 == 'none':
        print(q, random_rate, player, bet, random_action_used)
    return d0


def flop_decision(q, random_rate, player, bet):
    if np.random.uniform(0,1,1)>=random_rate:
        if player == 0:
            if bet == 'none':
                if q[[0,2]].max() == 0: #start with aggression
                    return 'bet'
                else:
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
                    if q[[1,3]].max()==0: #start with agression
                        return 'reraise'
                    else:
                        return ['call','reraise'][q[[1,3]].argmax()]
                else:
                    return 'fold'
    else:     
        if player == 0:
            if bet == 'none':
                return np.random.choice(['check','bet'])
            elif bet == 'bet':
                return 'call'
            elif bet == 'reraise':
                return 'call_reraise'
        elif player == 1:
            if bet == 'none':
                return np.random.choice(['check','bet'])
            elif bet == 'bet':
                return np.random.choice(['call','reraise'])
    
       
def simgame(q, gamma, df, random_rate):
    stage = 0 #0 is preflop, 1 is flop, 2 is turn, 3 is river
    player = 1 #blinds - p2 (idx=1) is dealer and therefore first to act
    bet = 0
    bet_type = 'none'
    bet_sizes = [2,4]
    game_over = False
    
    p1_val = [-2]
    p2_val = [-1]
    p1_stage = [0]
    p2_stage = [0]
    pot = 3
    
        
    #first action from dealer
    q_values = q[0][player, df['rank2'], :]
    action = preflop_decision(q_values, random_rate, player, bet_type)
    #if df['rank2'] == 3:
        #print('action:', action, 'q_vals:', q_values, random_rate, player, bet_type)
    p2_action = [action]
    p2_val_win = [pot]
    if action == 'bet':
        bet = bet_sizes[0]
        pot += bet + 1 #need to call small blind first
        p2_val[0] += -bet - 1
        bet_type = 'bet'
    elif action == 'fold':
        game_over = True
    elif action == 'check':
        p2_val[0] += -1 #call small blind
        pot += 1
        
    
    #first position's preflop decision (2nd total preflop action)
    if game_over:    
        p1_action = []
    else:
        player = 0 #p1 = index 0 = first to act
        q_values = q[0][player, df['rank1'], :]
        action = preflop_decision(q_values, random_rate, player, bet_type)
        p1_action = [action]
        p1_val_win = [pot]
        if action != 'bet' and action != 'reraise':
                       
            if action == 'call':
                pot += bet
                p1_val[0] += -bet
                p2_val_win[-1] += bet #if opp calls a bet and player wins, this value should be included in the bet
                pre_flop_bet = 1
            elif action == 'check':
                pre_flop_bet = 0
            elif action == 'fold':
                p2_val[0] = 2
                game_over = True
            
        else: #p1 bets/reraises
            bet = bet_sizes[0]
            if action == 'bet':
                bet_type = 'bet'
                p1_val[0] = p1_val[0] - bet
                pot += bet
            else:
                bet_type = 'reraise'
                p2_val_win[-1] += bet #if opp calls a bet and player wins, this value should be included in the bet
                p1_val[0] = p1_val[0] - bet * 3  #need to call p1's bet first and reraise is 2x bet
                pot += bet * 3
                bet = bet * 2 #double the bet on reraise
            
            #p2's respsone to p1's bet/reraise (their 2nd preflop decision, 3rd total preflop action)
            player = 1
            q_values = q[0][player, df['rank2'], :]
            action = preflop_decision(q_values, random_rate, player, bet_type)
            p2_action = p2_action + [action]
            p2_val_win += [pot]
            p2_stage = p2_stage + [0]
            
            if action == 'fold':
                game_over = True
                p1_val[-1] += pot
                p2_val += [0]
            else:
                p1_val_win[-1] += bet #if opp calls a bet and player wins, this value should be included in the bet
                p2_val = p2_val + [-bet]
                pot += bet
                if action == 'call':
                    pre_flop_bet = 2
                elif action == 'call_reraise':
                    pre_flop_bet = 3
                else:
                    raise Exception('Unexpected Action')
#xxxxxxxxxxxx Start of Flop coding  xxxxxxxxxxxx xxxxxxxxxxxx xxxxxxxxxxxx
    if not game_over:
        stage = 1        
        player = 0
        bet = 0
        bet_type = 'none'
        
        #first flop action - p1
        q_values = q[1][player, pre_flop_bet, df['p1_hand_value'], df['suitless1'], df['p1_sub_value'], :]
        action = flop_decision(q_values, random_rate, player, bet_type)
        
        p1_action = p1_action + [action]
        p1_val_win += [pot]
        p1_stage = p1_stage + [stage]
        
        if action == 'bet':
            bet = bet_sizes[stage]
            bet_type = 'bet'
            p1_val += [-bet]
            pot += bet 
            
        else:
            p1_val += [0]
        #p2's flop decision (2nd total flop action)
        if not game_over:    
            player = 1 #p2 = index 1 = dealer
            q_values = q[1][player, pre_flop_bet, df['p2_hand_value'], df['suitless2'], df['p2_sub_value'], :]
            action = flop_decision(q_values, random_rate, player, bet_type)
            p2_action = p2_action + [action]
            p2_val_win += [pot]
            p2_stage = p2_stage + [stage]
            if action =='check':
                p2_val += [0]
            elif action == 'call':
                pot += bet
                p2_val += [-bet]
                p1_val_win[-1] += bet #if opp calls a bet and player wins, this value should be included in the bet
                
            elif action == 'fold':
                p1_val[-1] += pot
                game_over = True
                p2_val += [0]
            elif action == 'bet' or action == 'reraise': #p2 bets/reraises
                bet = bet_sizes[stage]
                
                if action == 'bet':
                    bet_type = 'bet'
                    p2_val += [-bet] 
                    pot += bet
                elif action == 'reraise':
                    bet_type = 'reraise'
                    p1_val_win[-1] += bet #if opp calls a bet and player wins, this value should be included in the bet
                    p2_val += [-bet*3] #need to call p1's bet plus their bet
                    pot += bet * 3
                    bet = bet * 2

                #p1's respsone to p2's bet/reraise (their 2nd decision of the round, 3rd total round action)
                player = 0
                
                q_values = q[1][player, pre_flop_bet, df['p1_hand_value'], df['suitless1'], df['p1_sub_value'], :]
                action = flop_decision(q_values, random_rate, player, bet_type)
                
                p1_action = p1_action + [action]
                p1_val_win += [pot]
                p1_stage = p1_stage + [stage]
                
                if action == 'fold':
                    game_over = True
                    p2_val[-1] += pot
                    p1_val += [0]
                elif action == 'call' or action=='call_reraise':
                    p1_val += [-bet]
                    pot += bet
                    p2_val_win[-1] += bet #if opp calls a bet and player wins, this value should be included in the bet
                    
        if (action=='call' or action=='check' or action=='call_reraise'):
            #determine who win's the pot at showdown
            
            if df['result'] == 1:
                p1_val[-1] +=  pot
            elif df['result'] == -1 :
                p2_val[-1] += pot
            elif df['result'] == 0:
                p2_val[-1] += pot/2
                p1_val[-1] += pot/2
            else:
                raise Exception()
    
    if p1_action == []:
        p1_action = ['dealer_folds_preflop']
    
    if p1_val[-1]>0:
        p1_val = p1_val_win
    if p1_action != ['dealer_folds_preflop']:
        for idx in range(len(p1_action)):
            if p1_action[idx]!='fold':
                if p1_stage[idx] == 0:
                    q[0][0, df['rank1'], action_dict[p1_action[idx]]] = q[0][0, df['rank1'], action_dict[p1_action[idx]]] * (1 - gamma) + gamma * p1_val[idx]
                else:
                    q[1][0, pre_flop_bet, df['p1_hand_value'], df['suitless1'], df['p1_sub_value'], action_dict[p1_action[idx]]] = q[1][0, pre_flop_bet, df['p1_hand_value'], df['suitless1'], df['p1_sub_value'], action_dict[p1_action[idx]]] * (1-gamma) + gamma * p1_val[idx]
    
    if p2_val[-1]>0:
        p2_val = p2_val_win
    if p1_action != ['dealer_folds_preflop']:
        for idx in range(len(p2_action)):
            if p2_action[idx]!='fold':
                if p2_stage[idx] == 0:
                    q[0][1, df['rank2'], action_dict[p2_action[idx]]] = q[0][1, df['rank2'], action_dict[p2_action[idx]]] * (1 - gamma) + gamma * p2_val[idx]
                else:
                    q[1][1, pre_flop_bet, df['p2_hand_value'], df['suitless2'], df['p2_sub_value'], action_dict[p2_action[idx]]] = q[1][1, pre_flop_bet, df['p2_hand_value'], df['suitless2'], df['p2_sub_value'], action_dict[p2_action[idx]]] * (1-gamma) + gamma * p2_val[idx]
    
    #return q
    return q, p1_stage, p1_val, p1_action, p2_stage, p2_val, p2_action


action_dict = {'check':0,'call':1,'bet':2,'reraise':3,'call_reraise':3}


hand_count = 10000 #dont change, should be 10k
first_run = False#for very first run only, if just adding additional runs to existing q_matrix, 
#then set to False!

q1_filename = 'q4.npy'
q0_filename = 'q0.npy'

if first_run:
    raise Exception('Warning! Q_matrix will be overwritten if you proceed')

for simulation_number in range(2000): #increase this number for more (rather than hand_count)
    
    if first_run:
        df, q = generate_hands(hand_count, gen_q_matrix = True)
        first_run = False
        total_simulations = pd.read_csv('simulation_count.csv').iloc[0,0]
    else:
        df = generate_hands(hand_count, gen_q_matrix = False)
        
        if simulation_number == 0:
            
            pre = np.load(q0_filename)
            flop = np.load(q1_filename)
            q = [pre, flop]
            total_simulations = pd.read_csv('simulation_count.csv').iloc[0,0]
    print('Hands finished generating in {0:,.0f} seconds'.format(time.time()-start))      
    
    #Run Simulations
    res = pd.DataFrame(columns=['stage1','val1','act1','stage2','val2','act2'])
    for i in range(len(res), len(df)):
        if i%1000==0:
            print('Simulating Hand Number: {0:,.0f}'.format(i+simulation_number*hand_count))
    
        random_index = [idx for idx, x in enumerate([0, 5000, 10000, 20000]) if i>=x][-1]
        random_rates = [0.1, 0.1, 0.1, 0.1]
        #learning_rates = [0.333, 0.2, 0.15, 0.1]
        learning_rates = [0.05, 0.05, 0.05, 0.05]
        random_val = random_rates[random_index]
        gamma = learning_rates[random_index]
    
        q, p1_stage, p1_val, p1_action, p2_stage, p2_val, p2_action = simgame(q, gamma, df.iloc[i], random_val)
        resx = pd.DataFrame({'stage1':[p1_stage], 'val1':[p1_val], 'act1':[p1_action], 'stage2':[p2_stage], 'val2':[p2_val], 'act2':[p2_action]})
        res = res.append(resx)
        
    
    #error checking        
    for i, col in enumerate(['stage1','val1','act1','stage2','val2','act2'],1):
        res['len'+str(i)] = res[col].apply(lambda x: len(x))
        
    if sum((res['len1'] != res['len2']) | (res['len1'] != res['len3']) | (res['len4'] != res['len5']) | (res['len4'] != res['len6'])):
        raise Exception('expected length 0')
    
    res.drop(['len1','len2','len3','len4','len5','len6'], axis=1, inplace=True)
    res.reset_index(inplace=True, drop=True)
    
   
    df = pd.concat([df,res], axis=1)
    df['total2'] = df['val2'].apply(lambda x: sum(x) if x[-1]<=0 else x[-1])
    df['total1'] = df['val1'].apply(lambda x: sum(x) if x[-1]<=0 else x[-1])
    df['first_act1'] = df['act1'].apply(lambda x: x[0])
    df['first_act2'] = df['act2'].apply(lambda x: x[0])
    print('dealer fold precentage is {0:,.1f}%'.format(100*len(df[df['first_act1']=='dealer_folds_preflop'])/len(df)))
    
    
    r = [0, 1, 2, 4, 9, 16, 20, 28, 35, 45, 49, 50, 51]
    r =pd.DataFrame(r, columns=['rank1'])
    
    r['pocket_pair1'] = 1
    df = df.merge(r, how='left', on=['rank1'])
    
    if simulation_number%100==0:
        df.to_pickle('df{}.pkl'.format(simulation_number))    
    
    #pp = df[(df['pocket_pair1']==1)&(df['p1_hand_value']==1)][['total1','p1_sub_value','rank1']].groupby(['rank1','p1_sub_value']).agg(['mean','count'])

    pd.DataFrame([total_simulations + (simulation_number+1) * hand_count]).to_csv('simulation_count.csv', index=False)
    np.save(q0_filename, q[0])
    np.save(q1_filename, q[1])
    stop = time.time()
    print('{0:,.0f} seconds'.format(stop - start))