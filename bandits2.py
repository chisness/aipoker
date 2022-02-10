import random
import numpy as np
import matplotlib.pyplot as plt

class Arm:
	def __init__(self, n_pulls, total_reward):
		self.n_pulls = n_pulls
		self.total_reward = total_reward

	def average_reward(self):
		if self.n_pulls == 0:
			return 0
		return self.total_reward / self.n_pulls

# class Bandits - abstract superclass
# class EpsilonGreedy(Bandits)
# class UCB(Bandits)
# class ThompsonSampling(Bandits)

# concrete classes implement
# def choose_arm():


class Bandits:
	def __init__(self, arms, e=.1, n = 10):
		self.arms = arms
		self.e = e
		self.q = np.random.normal(0,1,self.arms)
		self.actual_best_arm = max(self.q)
		self.actual_best_arm_ix = np.argmax(self.q)
		self.n = n
		self.arm_history = { arm: Arm(0, 0) for arm in range(0, arms) }
		self.total_reward = 0
		self.total_pulls = 0

	def reset(self):
		pass

	def initial_pulls(self):
		for arm in range(0, self.arms):
			for i in range(0, n):
				self.pull(arm)

	def choose_arm(self):
		if random.random() < self.e:
			return random.randint(0, self.arms - 1)
		best_arm = self.best_arm()
		return best_arm

	# TODO: move everything that is common to all pull strategies into shared method
	def pull(self, arm_choice = None):
		arm_choice = self.choose_arm() if arm_choice is None else arm_choice
		reward = np.random.normal(self.q[arm_choice], 1)
		arm = self.arm_history[arm_choice]
		arm.n_pulls += 1
		arm.total_reward += reward
		self.total_reward += reward
		self.total_pulls += 1
		return reward

	def best_arm(self):
		best_arm_ix = None
		best_arm_reward_avg = -float('inf')
		for (arm_ix, arm) in self.arm_history.items():
			avg = arm.average_reward()
			if best_arm_reward_avg < avg:
				best_arm_reward_avg = avg
				best_arm_ix = arm_ix
		return best_arm_ix

	def average_reward(self):
		return self.total_reward/self.total_pulls

	def best_reward_total(self):
		return self.total_pulls*self.q[self.best_arm()]

	def average_regret(self):
		return ((self.total_pulls*self.actual_best_arm - self.total_reward) / (self.total_pulls))

	def average_optimalaction(self):
		pass

	#EPSILON-GREEDY with epsilon 
	#2000 runs each of 1000 time steps, take average of all


if __name__ == "__main__":
	# for n in [10, 100]:
	# 	b = Bandits(10, e=.1, n=n)
	# 	b.initial_pulls()
	# 	print(b.total_reward)
	# 	print('Average reward:', b.average_reward())
	# 	print('Best reward:', b.best_reward_total())
	# 	print('Best arm:', b.best_arm())
	# 	for i in range(0, 100):
	# 		b.pull()

		n = 10
		timesteps = 1000
		runs = 2000

		avg_rewarde000 = np.zeros((timesteps, runs))
		avg_rewarde001 = np.zeros((timesteps, runs))
		avg_rewarde010 = np.zeros((timesteps, runs))
		avg_rewarde050 = np.zeros((timesteps, runs))


		avg_regrete000 = np.zeros((timesteps, runs))
		avg_regrete001 = np.zeros((timesteps, runs))
		avg_regrete010 = np.zeros((timesteps, runs))
		avg_regrete050 = np.zeros((timesteps, runs))

		#b = Bandits(10, e=.1, n=n)
		for i in range(runs):
			b = Bandits(10, e=0, n=n)
			#b.initial_pulls()
			for j in range(timesteps):
				b.pull()
				avg_rewarde000[j,i] = b.average_reward()
				avg_regrete000[j,i] = b.average_regret()

		for i in range(runs):
			b = Bandits(10, e=0.01, n=n)
			#b.initial_pulls()
			for j in range(timesteps):
				b.pull()
				avg_rewarde001[j,i] = b.average_reward()
				avg_regrete001[j,i] = b.average_regret()

		for i in range(runs):
			b = Bandits(10, e=0.1, n=n)
			#b.initial_pulls()
			for j in range(timesteps):
				b.pull()
				avg_rewarde010[j,i] = b.average_reward()
				avg_regrete010[j,i] = b.average_regret()

		for i in range(runs):
			b = Bandits(10, e=0.5, n=n)
			#b.initial_pulls()
			for j in range(timesteps):
				b.pull()
				avg_rewarde050[j,i] = b.average_reward()
				avg_regrete050[j,i] = b.average_regret()
		#print(avg_reward)
		plt.figure(figsize=(15, 4), dpi=100)
		plt.plot(np.arange(timesteps),np.mean(avg_rewarde000, axis=1), color='green', label ='epsilon = 0 (greedy)')
		plt.plot(np.arange(timesteps),np.mean(avg_rewarde001, axis=1), color='red', label = 'epsilon = 0.01')
		plt.plot(np.arange(timesteps),np.mean(avg_rewarde010, axis=1), color='blue', label = 'epsilon = 0.1')
		plt.plot(np.arange(timesteps),np.mean(avg_rewarde050, axis=1), color='orange', label = 'epsilon = 0.5')
		plt.ylabel('Average reward')
		plt.xlabel('Steps')
		plt.legend(loc='lower left')
		plt.show()

		plt.figure(figsize=(15, 4), dpi=100)
		plt.plot(np.arange(timesteps),np.mean(avg_regrete000, axis=1), color='green', label ='epsilon = 0 (greedy)')
		plt.plot(np.arange(timesteps),np.mean(avg_regrete001, axis=1), color='red', label = 'epsilon = 0.01')
		plt.plot(np.arange(timesteps),np.mean(avg_regrete010, axis=1), color='blue', label = 'epsilon = 0.1')
		plt.plot(np.arange(timesteps),np.mean(avg_regrete050, axis=1), color='orange', label = 'epsilon = 0.5')
		plt.ylabel('Average regret')
		plt.xlabel('Steps')
		plt.legend(loc='lower left')
		plt.show()


