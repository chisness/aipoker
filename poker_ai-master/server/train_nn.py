import subprocess
import numpy as np
import pydash

ports = [['neural_net', 30000],['newuak_v2', 30001],['newuak_v3', 30002]]
i = 1
while True:
    seed = np.random.randint(0, 5)
    np.random.shuffle(ports)
    params = tuple([seed] + pydash.map_(ports, '0') + pydash.map_(ports, '1'))
    print("episode params:", params)
    try:
        output = subprocess.check_call('./dealer nn_train ./kuhn.limit.3p.game 3000 %s  %s %s %s -p %i,%i,%i >> ../training_logs/test_vs_nash_agents_10x3x9' % params, shell=True)
    except subprocess.CalledProcessError as e:
        print(e)
    # print("episode params:", params)    
    i = i + 1
