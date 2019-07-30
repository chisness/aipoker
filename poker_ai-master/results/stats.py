import os

import sys

filename = sys.argv[1]
seed = None

if len(sys.argv) > 2:
    seed = int(sys.argv[2])
file = None

current_seed = 0
final_score = {}

try:
    file = open(filename, 'r')
    for line in file:
        line = line.replace('\n', '')
        if 'Seed' in line:
            current_seed = int(line.split(':')[1])
        

        if seed is not None:    
            if current_seed != seed:
                continue


        if 'SCORE' in line:
            game = line.split(':')
            # print(game)
            score = game[1].split('|')
            players = game[2].split('|')

            for i, player in enumerate(players):
                # print(player, score[i], final_score)
                if player in final_score:
                    final_score[player] += int(score[i])
                else: 
                    final_score[player] = int(score[i]) 
        
    print(final_score)

except IOError:
    print('cannot open file')


finally:
    file.close()
