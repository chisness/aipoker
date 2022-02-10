# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:12:39 2020

@author: mthom
"""

from tkinter import Tk, Canvas, PhotoImage, Button, Label 
import tkinter.font as tkFont
from Generate_Hands_and_States import generate_hands
from poker_AI import dealer_ai_preflop_decision, dealer_ai_flop_first_decision, dealer_ai_flop_response, dealer_show_down
from poker_AI import first_position_ai_preflop_decisions, first_position_ai_flop_decision, first_position_show_down
import numpy as np
import pandas as pd

delay = True
show_no_matter_what = False

dealer = True
player_chips = 100
ai_chips = 100

rank_dict = {0:'2', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8', 7:'9', 8:'10', 9:'J', 10:'Q', 11:'K', 12:'A'}
suit_dict = {0:'C', 1:'S', 2:'H', 3:'D'}

pre_ai = np.load('q0.npy')
flop_ai = np.load('q4.npy')
q = [pre_ai, flop_ai]
q_normal = q.copy()

#For Aggressive Opponents
pre_ai_defensive = np.load('q0 - defense.npy')
flop_ai_defensive = np.load('q4 - defense.npy')
q_aggressive = [pre_ai_defensive, flop_ai_defensive]


player_aggression = []


try:
    player_result = pd.read_csv('results.csv')
    player_chips = player_result.iloc[-1].iloc[0]
    ai_chips = 200 - player_chips
except FileNotFoundError:
    player_result = pd.DataFrame([100], columns=['player_chips'])


def deal():
    global player_chips, ai_chips, pot, btn_1, btn_2, btn_3, face_downimg, hole_img, hole_img2, ai_hole1, ai_hole2, deck1, deck2, deck3, df
    global flopimg1, flopimg2, flopimg3, oppimg1, oppimg2
    global player_aggression, current_aggression, q, skip_aggression
    
    skip_aggression = False
    btn_2.configure(text='', bg='green', bd=0, state='disabled')
    current_aggression = 0
    #print(player_aggression)
    
    if len(player_aggression)>5:
        raise Exception('Error, expecting information on last 5 hands only')
    
    #Determine whether to play a defensive AI if player has recently been betting aggressively
    if sum(player_aggression)>=10:
        q = q_aggressive.copy()
       #print('playing defensive ai\n')
    else:
        q = q_normal.copy()
        #print('playing regular ai\n')
    try: 
        canvas.delete(deck1)
        canvas.delete(deck2)
        canvas.delete(deck3)
    except NameError:
        pass
    
    #clear last game's results
    result_label.config(text='')
    
    df = generate_hands(1, False)
    df = df.T.squeeze()
    for i in range(4):
        df['p{}_c{}'.format(i//2+1,i%2+1)] = rank_dict[df['ranks'][i]] + suit_dict[df['suits'][i]]

    for i in range(1,4):
        df['flop{}'.format(i)] = rank_dict[df['ranks'][i+3]] + suit_dict[df['suits'][i+3]]

    if dealer:
        h1, h2 = df[['p2_c1', 'p2_c2']]
        opp1, opp2 = df[['p1_c1', 'p1_c2']]
    else:
        h1, h2 = df[['p1_c1', 'p1_c2']]
        opp1, opp2 = df[['p2_c1', 'p2_c2']]
        
    flop1, flop2, flop3 = df[['flop1','flop2','flop3']]
    
    #antes
    if dealer:
        player_chips += -1
        ai_chips += -2
    else:
        player_chips += -2
        ai_chips += -1
    pot = 3
    
    #get pictures
    hole_img = PhotoImage(file='./PNG/{}.png'.format(h1))
    hole_img2 = PhotoImage(file='./PNG/{}.png'.format(h2))
    
    flopimg1 = PhotoImage(file='./PNG/{}.png'.format(flop1))
    flopimg2 = PhotoImage(file='./PNG/{}.png'.format(flop2))
    flopimg3 = PhotoImage(file='./PNG/{}.png'.format(flop3))
    
    oppimg1 = PhotoImage(file='./PNG/{}.png'.format(opp1))
    oppimg2 = PhotoImage(file='./PNG/{}.png'.format(opp2)) 
    
    #AI's cards facedown
    ai_hole1 = canvas.create_image(20,100, anchor='nw', image=face_downimg)    
    ai_hole2 = canvas.create_image(200,100, anchor='nw', image=face_downimg)    
    
    #hole cards
    canvas.create_image(20,600, anchor='nw', image=hole_img)      
    canvas.create_image(200,600, anchor='nw', image=hole_img2)

    #update chip amounts
    ai_chips_label.configure(text=ai_chips)  
    player_chips_label.configure(text=player_chips)
    pot_label.configure(text='{}'.format(pot))
    
    if dealer:
        btn_1.configure(text='Bet (1 + 2)', state='active', bd=1, bg='gray', activebackground='gray', command=lambda: dealer_first_action('bet'))
        btn_2.configure(text='Call Small Blind (1)', state='active', bd=1, bg='gray', activebackground='gray',  command=lambda: dealer_first_action('check'))
        btn_3.configure(text='Fold', state='active', bd=1, bg='gray', activebackground='gray', command=lambda: dealer_first_action('fold'))
    else:
        first_position_ai_first_action()


#xxxxxxxxxxxxxxxxxxxxxxxxxxx AI is Dealer xxxxxxxxxxxxxxxxxxxxxxxxxxx
def first_position_ai_first_action(): #determine AI's first preflop decision
    global player_chips, ai_chips, pot, skip_aggression
    player_chips, ai_chips, pot, action = first_position_ai_preflop_decisions(q, df, player_chips, ai_chips, pot, 'none')

     #AI's preflop decision
    if delay:
        root.after(100, first_position_show_ai_decision, action)
    else:
        first_position_show_ai_decision(action)

def first_position_show_ai_decision(action): #show AI's first preflop decision
    global ai_chips, player_chips, pot
    global ai_chip_label, player_chips_label, pot_label
    global result_label, btn_1, btn_2, btn_3
    global current_aggression, skip_aggression
    
    ai_chips_label.configure(text=ai_chips)  
    player_chips_label.configure(text=player_chips)
    pot_label.configure(text=pot)
    
    
    
    if action == 'check': #Determine Player's Possible Preflop Options 
        result_label.configure(text='AI Checks Preflop')
        btn_1.configure(text='Bet (2)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_preflop_player_action('bet'))
        btn_2.configure(text='Check', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_preflop_player_action('check'))
        btn_3.configure(text='', state='disable', bd=0, bg = 'green', activebackground = 'green')
    elif action == 'fold':
        result_label.configure(text='AI Folds!')
        skip_aggression = True
        reset_game()
        current_aggression += 1
    elif action == 'bet':
        result_label.configure(text='AI Bets 2')
        btn_1.configure(text='Reraise (2 + 4)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_preflop_player_action('reraise'))
        btn_2.configure(text='Call (2)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_preflop_player_action('call'))
        btn_3.configure(text='Fold', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_preflop_player_action('fold'))
    else:
        raise Exception('Unexpected AI Action')

def first_position_preflop_player_action(action): #Process Player's Preflop Decision
    #process player's action
    global pot, player_chips, ai_chips
    global pot_label, player_chips_label, ai_chips_label
    global pre_flop_bet, btn_1, btn_2, btn_3
    global current_aggression
    
    player_fold = False
    
    btn_1.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    btn_2.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    btn_3.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    
    if action == 'fold':
        ai_chips += pot
        result_label.configure(text='You Folded')
        #current_aggression += -1
        reset_game() 
        player_fold = True
    elif action == 'call':
        player_chips += -2
        pot += 2
        pre_flop_bet = 1
    elif action == 'check':
        pre_flop_bet = 0
    elif action == 'bet':
        player_chips += -2
        pot += 2
        current_aggression += 1
    elif action == 'reraise':
        player_chips += -6
        pot += 6
        current_aggression += 2
    else:
        raise Exception('Unexpected Player Action')
    
    pot_label.config(text=pot)
    player_chips_label.config(text=player_chips)
    ai_chips_label.config(text=ai_chips)
    
    #AI's response to a bet/raise from player
    ai_decision = False
    if action == 'bet' or action == 'reraise':
        player_chips, ai_chips, pot, action = first_position_ai_preflop_decisions(q, df, player_chips, ai_chips, pot, action)
        ai_decision = True
                
    if not player_fold: #Show Flop / Or Show that AI Folded
        if delay:
            root.after(300, first_position_show_flop, action, ai_decision)
        else:
            first_position_show_flop(action, ai_decision)

def first_position_show_flop(action, ai_decision): #Process AI's Preflop Response
    global pot, player_chips, ai_chips
    global pot_label, player_chips_label, ai_chips_label
    global pre_flop_bet, current_aggression
    
    if ai_decision:
        if action == 'fold':
            result_label.configure(text='AI Folds!')
            current_aggression += 1
            reset_game() 
        
        elif action == 'call':
            result_label.configure(text='AI Calls')
            pre_flop_bet = 2
            
        elif action == 'call_reraise':
            result_label.configure(text='AI Calls Reraise')
            pre_flop_bet = 3
        else:
            raise Exception('Unexpected AI Action: {}'.format(action))
    
    
        pot_label.config(text=pot)
        player_chips_label.config(text=player_chips)
        ai_chips_label.config(text=ai_chips)
    
    if action != 'fold':
        if delay:
            root.after(100, show_flop1)
            root.after(150, show_flop2)
            root.after(200, show_flop3)
        else:
            show_flop1()
            show_flop2()
            show_flop3()
        
def first_position_show_flop_buttons(): #Populate Player's Flop Options (First Decision)
    global btn_1, btn_2, btn_3
    btn_1.configure(text='Bet (4)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_flop('bet'))
    btn_2.configure(text='Check', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_flop('check'))
    btn_3.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    result_label.config(text='')
    
    
def first_position_flop(action): #process player's first action on the flop (1st flop action overall)
    global player_chips, ai_chips, pot, player_chips_label, ai_chips_label, pot_label
    global btn_1, btn_2, btn_3, current_aggression
    
    btn_1.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    btn_2.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    btn_3.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    
    if action == 'check':
        bet_type = 'none'
    elif action == 'bet':
        player_chips += -4
        pot += 4
        bet_type = 'bet'
        current_aggression += 1
    else:
        raise Exception('Unexpected Player Action')
    
    ai_chips_label.configure(text=ai_chips)  
    player_chips_label.configure(text=player_chips)
    pot_label.configure(text=pot)
    
    #Determine AI's Flop Decision
    player_chips, ai_chips, pot, action = first_position_ai_flop_decision(q, pre_flop_bet, df, player_chips, ai_chips, pot, bet_type, random_rate=0)
    
    if delay:
        root.after(100, first_position_ai_flop_response, action)
    else:
        first_position_ai_flop_response(action)

#Show AI's Flop Decision
def first_position_ai_flop_response(action):#print result label for a brief period (in case AI calls/checks, so it doesn't immediately print showdown results)
    global result_label, pot, ai_chips, player_chips, pot_label, ai_chips_label, player_chips_label
    
    pot_label.config(text=pot)
    ai_chips_label.config(text=ai_chips)
    player_chips_label.config(text=player_chips)
    
    
    if action == 'fold':
        result_label.configure(text='AI Folded!')
        reset_game()
    elif action == 'check':
        result_label.configure(text='AI Checks')
    elif action == 'call':
        result_label.configure(text='AI Calls')
    elif action == 'bet':
        result_label.configure(text='AI Bets')
    elif action == 'reraise':
        result_label.configure(text='AI Reraises')     
    else:
        raise Exception('Unexpected AI Action')
    
    if action == 'bet' or action == 'reraise': #AI bets/reraises - Now Player Needs to Respond:
        first_position_flop_response(action)
    elif action == 'check' or action == 'call':
        if delay:
            root.after(300, first_position_immediate_showdown, action)
        else:
            first_position_immediate_showdown(action)
                
def first_position_flop_response(action): #update action buttons for Player's 2nd Flop Action
    #action from ai
    global result_label, ai_chips, player_chips, pot
    global player_chips_label, ai_chips_label
    global ai_bet_type
    
    ai_bet_type = ''
    
    player_chips_label.configure(text=player_chips)
    ai_chips_label.configure(text=ai_chips)
    pot_label.configure(text=pot)
    
    if action == 'bet':
        btn_1.configure(text='', state='disable', bd=0, bg = 'green', activebackground = 'green')
        btn_2.configure(text='Call (4)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_final_player_decision('call'))
        btn_3.configure(text='fold', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_final_player_decision('fold'))
        ai_bet_type = 'bet'
    elif action == 'reraise':
        btn_1.configure(text='', state='disable', bd=0, bg = 'green', activebackground = 'green')
        btn_2.configure(text='Call Reraise (8)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_final_player_decision('call_reraise'))
        btn_3.configure(text='fold', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: first_position_final_player_decision('fold'))
        ai_bet_type = 'reraise'

def first_position_immediate_showdown(action): #Showdown if AI Checks/Calls
    global result_label, pot, ai_chips, player_chips
    global result_label
    
    player_chips, ai_chips, result_text, show_ai_cards = first_position_show_down(df, pot, ai_chips, player_chips, False)
    result_label.config(text=result_text)
    if show_ai_cards:
        canvas.itemconfig(ai_hole1, image = oppimg1)
        canvas.itemconfig(ai_hole2, image = oppimg2)
    else:
        pass #AI Mucks
    reset_game()

def first_position_final_player_decision(action): #Player's Flop Response to a Bet/Reraise
    global result_label, pot, ai_chips, player_chips
    global btn_1, btn_2, btn_3, current_aggression, ai_bet_type
    
    btn_1.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    btn_2.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    btn_3.configure(text='', state='disabled', bd=0, bg = 'green', activebackground = 'green')
    
    if action == 'fold':
        result_label.config(text='You Folded')
        ai_chips += pot
        if ai_bet_type == 'bet':
            current_aggression += -1
        elif ai_bet_type == 'reraise':
            current_aggression += -2
        else:
            raise Exception('Was expecting bet or reriase ofr ai_bet_type')
        ai_bet_type = ''
    else:
        if action == 'call':
            player_chips += -4
            pot += 4
        elif action == 'call_reraise':
            player_chips += -8
            pot+= 8
            current_aggression += 2
        else:
            raise Exception('Unexpected Player Action')
        
        player_chips, ai_chips, result_text, show_ai_cards = first_position_show_down(df, pot, ai_chips, player_chips, True)
    
        result_label.config(text=result_text)
    
        if show_ai_cards:
            canvas.itemconfig(ai_hole1, image = oppimg1)
            canvas.itemconfig(ai_hole2, image = oppimg2)
        else:
            pass #AI Mucks
   
    reset_game()
#xxxxxxxxxxxxxxxxxxxxxxxxxxx END xxxxxxxxxxxxxxxxxxxxxxxxxxx
        
    
#xxxxxxxxxxxxxxxxxxxxxxxxxxx Player is Dealer xxxxxxxxxxxxxxxxxxxxxxxxxxx
def dealer_first_action(action):
    global pot, ai_chips, player_chips 
    global pot_label, ai_chips_label, player_chips_label
    global btn_1, btn_2, btn_3, current_aggression, skip_aggression
    
    #disable decisions
    btn_1.configure(text='', bg='green', bd=0, state='disabled')
    btn_2.configure(text='', bg='green', bd=0, state='disabled')
    btn_3.configure(text='', bg='green', bd=0, state='disabled')
    
    pot = 3
    ai_chips_label.configure(text=ai_chips)  
    player_chips_label.configure(text=player_chips)
    pot_label.configure(text=pot)
        
    #player's action
    if action == 'bet':
        player_chips += -3 #call small blind + bet of 2
        pot += 3
        player_chips_label.configure(text=player_chips)
        pot_label.configure(text=pot)
        current_aggression += 1
    elif action == 'check':
        player_chips += -1 #call small blind
        pot += 1
    elif action == 'fold':
        ai_chips += pot
        skip_aggression = True
        result_label.configure(text='You Folded')
        reset_game()        
        
    #update chip amounts
    ai_chips_label.configure(text=ai_chips)  
    player_chips_label.configure(text=player_chips)
    pot_label.configure(text=pot)
    


    #if player doesn't fold
    if action != 'fold': #AI's preflop decision
        player_chips, ai_chips, pot, action = dealer_ai_preflop_decision(q, df, player_chips, ai_chips, pot, action)
        
        #ai's preflop decision
        if delay:
            root.after(100, dealer_show_ai_decision, action)
        else:
            dealer_show_ai_decision(action)
            
        
def dealer_show_ai_decision(action):
    global ai_chips, player_chips, pot
    global ai_chip_label, player_chips_label, pot_label
    global result_label, btn_1, btn_2, btn_3
    global pre_flop_bet, current_aggression, ai_bet_type
    
    ai_chips_label.configure(text=ai_chips)  
    player_chips_label.configure(text=player_chips)
    pot_label.configure(text=pot)
    
    ai_bet_type = ''
    
    if action == 'check':
        result_label.configure(text='AI Checks Preflop')
        pre_flop_bet = 0
    elif action == 'call':
        result_label.configure(text='AI Calls Preflop')
        pre_flop_bet = 1
    elif action == 'fold':
        result_label.configure(text='AI Folds!')
        current_aggression += 1
        reset_game()
    elif action == 'bet':
        ai_bet_type = 'bet'
        result_label.configure(text='AI Bets 2')
        btn_1.configure(text='', bg='green', bd=0, state='disabled')
        btn_2.configure(text='Call (2)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: dealer_preflop_response('call'))
        btn_3.configure(text='Fold', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: dealer_preflop_response('fold'))
    elif action== 'reraise':
        ai_bet_type = 'reraise'
        result_label.configure(text='AI Reraises!')
        btn_1.configure(text='', bg='green', activebackground='green', bd=0, state='disabled')
        btn_2.configure(text='Call Reraise (4)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: dealer_preflop_response('call_reraise'))
        btn_3.configure(text='Fold', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: dealer_preflop_response('fold'))        
    else:
        raise Exception('Unexpected AI Action')
    
    if action == 'check' or action == 'call':        
        if delay:
            show_flop1()
            root.after(50, show_flop2)
            root.after(100, show_flop3)
        else:
            show_flop1()
            show_flop2()
            show_flop3()


def dealer_preflop_response(action):
    global ai_chips, player_chips, pot, result_label, btn_1, btn_2, btn_3
    global pre_flop_bet, current_aggression, ai_bet_type
    
    btn_1.configure(text='', bg='green', bd=0, state='disabled')
    btn_2.configure(text='', bg='green', bd=0, state='disabled')
    btn_3.configure(text='', bg='green', bd=0, state='disabled')
            
    result_label.configure(text='')
    
    if action == 'fold':
        ai_chips += pot
        result_label.configure(text='You Folded')
        reset_game()
        if ai_bet_type == 'bet':
            current_aggression += -1
        elif ai_bet_type == 'reraise':
            current_aggression += -2
        else:
            raise Exception('Unexpected ai_bet_type: {}'.format(ai_bet_type))
        ai_bet_type = ''
    elif action == 'call':
        pre_flop_bet = 2
        pot += 2
        player_chips += -2
    elif action == 'call_reraise':
        pot += 4
        pre_flop_bet = 3
        player_chips += -4
        current_aggression += 2
    else:
        raise Exception('Unexpected Player Action: {}'.format(action))

    ai_chips_label.configure(text=ai_chips)  
    player_chips_label.configure(text=player_chips)
    pot_label.configure(text=pot)
    
    if action != 'fold':
        if delay:
            root.after(0, show_flop1)
            root.after(50, show_flop2)
            root.after(100, show_flop3)
        else:
            show_flop1()
            show_flop2()
            show_flop3()
            
            
def show_flop1():
    global deck1
    deck1 = canvas.create_image(20,350, anchor='nw', image=flopimg1)    

def show_flop2():
    global deck2    
    deck2 = canvas.create_image(200,350, anchor='nw', image=flopimg2)    

def show_flop3():
    global deck3, result_label
    deck3 = canvas.create_image(380,350, anchor='nw', image=flopimg3)   
    if dealer:
        if delay:
            root.after(200, dealer_ai_first_flop_decision)
        else:
            root.after(0, dealer_ai_first_flop_decision)
    else:
        first_position_show_flop_buttons()
         

def dealer_ai_first_flop_decision():
    global result_label, pot, ai_chips, player_chips 
    global btn_1, btn_2, btn_3
    global ai_chips_label, player_chips_label, pot_label

    result_label.configure(text='')
    ai_chips, pot, action = dealer_ai_flop_first_decision(q, pre_flop_bet, df, pot, ai_chips) 
    
    ai_chips_label.configure(text=ai_chips)  
    player_chips_label.configure(text=player_chips)
    pot_label.configure(text=pot)
    
    if action == 'check':
        btn_1.configure(text='Bet (4)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: dealer_flop_decision('bet'))
        btn_2.configure(text='Check', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: dealer_flop_decision('check'))
        btn_3.configure(text='', bg='green', activebackground='green', bd=0, state='disabled')
        result_label.configure(text='AI Checks the Flop')
    elif action == 'bet':
        btn_1.configure(text='Reraise (4 + 8)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: dealer_flop_decision('reraise'))
        btn_2.configure(text='Call (4)', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: dealer_flop_decision('call'))
        btn_3.configure(text='Fold', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: dealer_flop_decision('fold'))
        result_label.configure(text='AI Bets the Flop')
    else:
        raise Exception('Unexpected AI action')


def dealer_flop_decision(action):
    global ai_chips, player_chips, pot, result_label
    global btn_1, btn_2, btn_3, current_aggression
    
    #disable decisions
    btn_1.configure(text='', bg='green', bd=0, state='disabled')
    btn_2.configure(text='', bg='green', bd=0, state='disabled')
    btn_3.configure(text='', bg='green', bd=0, state='disabled')
    
    player_fold = False
    player_call = False
    
    #Players Flop Decision:
    if action == 'fold':
        current_aggression += -1
        player_fold = True
        ai_chips += pot
        result_label.configure(text='You Folded')
        reset_game()                
    elif action == 'check':
        player_call = True
    elif action == 'call':
        player_call = True
        player_chips += -4
        pot += 4
    elif action == 'bet':
        player_chips += -4
        pot += 4
        current_aggression += 1
        pot, ai_chips, player_chips, action = dealer_ai_flop_response(q, pre_flop_bet, df, pot, ai_chips, player_chips, 'bet')
    elif action == 'reraise':
        player_chips += -12
        pot += 12
        current_aggression += 2
        pot, ai_chips, player_chips, action = dealer_ai_flop_response(q, pre_flop_bet, df, pot, ai_chips, player_chips, 'reraise')
    else:
        raise Exception('Unexpected Player Action: {}'.format(action))
    
    if player_fold == False:
        if delay:
            root.after(100, dealer_show_ai_final_flop_decision, action, player_call)
        else:
            dealer_show_ai_final_flop_decision(action, player_call)
    else:
        reset_game()
    
   
def dealer_show_ai_final_flop_decision(action, player_call):
    global ai_chips, player_chips, pot, result_text, result_label
    if action != 'fold': #showdown
        player_chips, ai_chips, result_text, show_ai_cards = dealer_show_down(df, pot, ai_chips, player_chips, player_call)
        result_label.configure(text=result_text)
        if show_ai_cards:
            canvas.itemconfig(ai_hole1, image = oppimg1)
            canvas.itemconfig(ai_hole2, image = oppimg2)
        else:
            #indicate that AI mucks card
            pass
        
    else:
        result_label.configure(text='AI Folded!')
    
    reset_game()

               
def reset_game():
    global pot, player_chips, ai_chips
    global btn_1, btn_2, btn_3, dealer_btn, pot_window, pot_txt_window, dealer, ai_chips_label, player_chips_label, pot_label
    global player_result, player_aggression, current_aggression, skip_aggression
    
    
    #update regression for recent experience (dont add a 0 if AI is dealer and immediately folds)
    if not skip_aggression:
        player_aggression = [current_aggression] + player_aggression[0:4]
    
    #dealer = bool(1-dealer)
    dealer = bool(1-dealer)
    
    #configure buttons for new game
    btn_1.configure(text='', bg='green', bd=0, state='disabled')
    btn_2.configure(text='Deal', state='active', bd=1, bg = 'gray', activebackground = 'gray', command=lambda: deal())
    btn_3.configure(text='', bg='green', bd=0, state='disabled')
    
    #Record Hand Results
    player_result.loc[len(player_result)] = player_chips
    player_result.to_csv('results.csv', index=False)
    print('You have played {0:,.0f} hands'.format(len(player_result)))
    #update chip amounts
    ai_chips_label.configure(text=ai_chips)  
    player_chips_label.configure(text=player_chips)
    pot_label.configure(text=pot)
    #move dealer button and pot
    canvas.delete(dealer_btn)
    
    if dealer:
        dealer_btn = canvas.create_image(650,650, anchor='nw', image=dealer_btn_img)
        pot_window = canvas.create_window(715, 180, anchor='nw', window=pot_label)
        pot_txt_window = canvas.create_window(640, 196, anchor='nw', window=pot_text)
    else:
        dealer_btn = canvas.create_image(650,140, anchor='nw', image=dealer_btn_img)
        pot_window = canvas.create_window(715, 694, anchor='nw', window=pot_label)
        pot_txt_window = canvas.create_window(640, 710, anchor='nw', window=pot_text)
     
    if show_no_matter_what:
        canvas.itemconfig(ai_hole1, image = oppimg1)
        canvas.itemconfig(ai_hole2, image = oppimg2)
    
    
if __name__ == '__main__':
    root = Tk()      
    root.title('Poker')
    #root.configure(bg='green')
    canvas = Canvas(root, width = 1000, height = 1000, bg='green')      
    canvas.grid()      
    
    face_downimg = PhotoImage(file='./PNG/blue_back.PNG')
    dealer_btn_img = PhotoImage(file='./PNG/dealer_button.PNG')
    
    #dealer button
    if dealer:
        dealer_btn = canvas.create_image(650,650, anchor='nw', image=dealer_btn_img)
    else:
        dealer_btn = canvas.create_image(650,140, anchor='nw', image=dealer_btn_img)
    
   #"deck"
    deck = canvas.create_image(700,350, anchor='nw', image=face_downimg)    
    
    #Action buttons
    btn_1 = Button(canvas, text='', width=15, height=3, bg='green', bd=0, state='disabled')
    btn_1_window = canvas.create_window(550, 610, anchor='ne', window=btn_1)
    
    btn_2 = Button(canvas, text='Deal', width=15, height=3, bg='gray', activebackground='gray', command=deal)
    btn_2_window = canvas.create_window(550, 690, anchor='ne', window=btn_2)
    
    btn_3 = Button(canvas, text='', width=15, height=3, bg='green', bd=0, state='disabled')
    btn_3_window = canvas.create_window(550, 770, anchor='ne', window=btn_3)
    
    
    #chip labels
    labelFontStyle = tkFont.Font(size=20, weight='bold')
    numberFontStyle = tkFont.Font(size=14, weight='bold')
    
    player_chips_label = Label(canvas, text=player_chips, width = 15, height = 3)#, font=numberFontStyle)#, bg='green')
    player_chip_window = canvas.create_window(330, 860, anchor='ne', window=player_chips_label)
    
    player_chip_txt = Label(canvas, text='Player\'s chips:', bg='green', font=numberFontStyle)
    player_chip_txt_window = canvas.create_window(50, 870, anchor='nw', window=player_chip_txt)
    
    ai_chips_label = Label(canvas, text=ai_chips, width = 15, height = 3)#, font=numberFontStyle)#, bg='green')
    ai_chip_window = canvas.create_window(330, 30, anchor='ne', window=ai_chips_label)
    
    ai_chip_txt = Label(canvas, text='AI\'s chips:', bg='green', font=numberFontStyle)
    ai_chip_txt_window = canvas.create_window(80, 40, anchor='nw', window=ai_chip_txt)
    
    #pot
    pot_label = Label(canvas, text='0', width = 10, height = 3, font=numberFontStyle)
    pot_text = Label(canvas, text='Pot:', bg='green', font=labelFontStyle )
    
    #result message
    resultFont = tkFont.Font(size=20, weight='bold')
    result_label = Label(canvas, text='', width = 30, height = 3, font=resultFont, bg='green', fg='red')
    result_window = canvas.create_window(400, 5, anchor='nw', window=result_label)
    
    if dealer:
        pot_window = canvas.create_window(715, 180, anchor='nw', window=pot_label)
        pot_txt_window = canvas.create_window(640, 196, anchor='nw', window=pot_text)
    else:
        pot_window = canvas.create_window(715, 694, anchor='nw', window=pot_label)
        pot_txt_window = canvas.create_window(640, 710, anchor='nw', window=pot_text)
        
    root.mainloop()   