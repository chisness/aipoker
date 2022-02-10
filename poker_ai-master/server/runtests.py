import os 
import itertools
import subprocess
import sys
from multiprocessing import Pool
agent_list = ['newuak.sh']

path = os.getcwd()

for agent in os.listdir(path):
    if '.sh' in agent and agent != 'kuhn.sh' and agent != 'newuak.sh' and agent != 'keepalive.sh':
        agent_list.append(agent)

print(agent_list)
version = 'v2'

# print(sys.argv)

# if '-v' in sys.argv:
#     version = sys.argv[2]

# matches = list(itertools.combinations(agent_list, 3))

# print(len(matches))

# if '-f' not in sys.argv:
#     matches = list(filter(lambda x: 'newuak.sh' in x, matches))

# # matches = [('olduak.sh', 'efs.sh', 'juan.sh')]
# matches = [('newuak.sh', 'bayes.sh', 'te.sh')]
# print(len(matches))



# def run_matches(match):
#     params = match + (version,)
#     print('params: ', params)
#     try:
#         output = subprocess.check_call('./kuhn.sh 9 100 %s %s %s %s' % params, shell=True)
#         print(output)
#     except subprocess.CalledProcessError as e:
#         print('hello')
#         print(e)


# if __name__ == '__main__':
#     pool = Pool()
#     result = pool.map(run_matches, matches)
