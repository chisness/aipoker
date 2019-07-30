import os 
import itertools
import subprocess
import sys
from multiprocessing import Pool
agent_list = ['newuak.sh']

for agent in os.listdir():
    if '.sh' in agent and agent != 'kuhn.sh' and agent != 'newuak.sh' and != 'keepalive.sh':
        agent_list.append(agent)


version = 'v2'

if '-v' in sys.argv:
    version = sys.argv[2]

matches = list(itertools.combinations(agent_list, 3))

if '-f' not in sys.argv:
    matches = list(filter(lambda x: 'newuak.sh' in x, matches))

matches = [('olduak.sh', 'efs.sh', 'juan.sh')]
print(len(matches))



def run_matches(match):
    params = match + (version,)
    print(params)
    try:
        output = subprocess.check_call('./kuhn.sh 9 3000 %s %s %s %s' % params, shell=True)
    except subprocess.CalledProcessError as e:
        print(e)


if __name__ == '__main__':
    pool = Pool()
    result = pool.map(run_matches, matches)
