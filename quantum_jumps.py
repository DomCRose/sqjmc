import numpy as np
from scipy import linalg

class jump_trajectory_generator(object):

	"""Generates quantum jump trajectories using a binary search.

	"""

	def __init__(self, lindbladian, smallest_time, evolver_number):
		self.model = lindbladian
		self.smallest_time = smallest_time
		self.evolver_number = evolver_number
		self._effective_hamiltonian()
		self._binary_evolution_operators()

	def _effective_hamiltonian(self):
		"""The non-hermitian generator for evolution between jumps."""
		self.effective_hamiltonian = np.array(self.model.hamiltonian)
		for jump in self.model.jump_operators:
			self.effective_hamiltonian -= 0.5j * np.conjugate(jump).T @ jump

	def _binary_evolution_operators(self):
		"""Evolution operators for the binary search of jump times."""
		smallest_evolver = linalg.expm(-1j*self.smallest_time*self.effective_hamiltonian)
		evolvers = [smallest_evolver]
		for i in range(self.evolver_number - 1):
			evolvers.append(evolvers[-1] @ evolvers[-1])
		self.evolver_steps = [2**(self.evolver_number - 1 - i) 
							  for i in range(self.evolver_number)]
		self.evolvers = evolvers[::-1]

	def _evolver_combination(self, steps):
		"""Calculates the combination of evolvers to reach the next observation."""
		combination = []
		for evolver_step in self.evolver_steps:
			combination.append(int(steps / evolver_step))
			steps = steps % evolver_step
			if steps == 0:
				combination.extend([
					0 for i in range(len(self.evolver_steps) - len(combination))])
				break
		return combination

	def _jump_search(self, evolver_index):
		self.current_state = np.dot(self.evolvers[evolver_index+1], self.previous_state)
		probability = linalg.norm(self.current_state)**2
		for i in range(evolver_index+2, len(self.evolvers)):
			if probability > self.random:
				self.current_steps += self.evolver_steps[i - 1]
				self.previous_state = np.array(self.current_state)
				self.current_state = np.dot(self.evolvers[i], self.previous_state)
			else:
				self.current_state = np.dot(self.evolvers[i], self.previous_state)
			probability = linalg.norm(self.current_state)**2
		if probability > self.random:
			self.current_steps += self.evolver_steps[-1]
		else:
			self.current_state = self.previous_state

	def _jump(self):
		jump_states = np.dot(self.model.jump_operators, self.current_state)
		jump_probabilities = np.linalg.norm(jump_states, axis = 1)**2
		jump_probabilities = jump_probabilities/sum(jump_probabilities)
		stacked_probabilities = [sum(jump_probabilities[:(index+1)]) 
								 for index in range(len(jump_probabilities))]
		jump_random = np.random.random()
		jump_index = np.searchsorted(stacked_probabilities, jump_random, side = 'right')
		self.current_state = jump_states[jump_index]/linalg.norm(jump_states[jump_index])

	def _binary_evolution_step(self, evolver_index, evolutions):
		for step in range(evolutions):
			if self.probability <= self.random:
				self._jump_search(evolver_index)
				self._jump()
				self.random = np.random.random()
				self.jump_occurred = True
			self.current_steps += self.evolver_steps[evolver_index] 
			self.previous_state = np.array(self.current_state)
			self.current_state = np.dot(self.evolvers[evolver_index], self.previous_state)
			self.probability = linalg.norm(self.current_state)**2

	def _binary_evolution(self, steps):
		self.current_steps = 0
		self.jump_occurred = False
		combination = self._evolver_combination(steps)
		for evolver_index in range(self.evolver_number):
			if combination[evolver_index] >= 1:
				self.current_state = np.dot(self.evolvers[evolver_index], 
											self.previous_state)
				self.probability = linalg.norm(self.current_state)**2
				self._binary_evolution_step(evolver_index, combination[evolver_index]-1)
			if self.jump_occurred:
				self._binary_evolution(steps-self.current_steps)
				self.jump_occurred = False
				break
			self.previous_state = self.current_state

	def trajectory(self, state, observation_number, steps_per_observation, 
				   observation, *observation_args):
		self.current_state = state
		self.previous_state = state
		self.random = np.random.random()
		results = [observation(self.current_state, *observation_args)]
		for i in range(observation_number):
			if i % int(observation_number / 10) == 0:
				print("Progress: " + str(int(i*100/observation_number)) + "%")
			self._binary_evolution(steps_per_observation)
			results.append(observation(self.current_state/linalg.norm(self.current_state),
									   *observation_args))
		print("Progress: 100%")
		return np.array(results)

	def stochastic_average(self, state, observation_number, steps_per_observation, 
						   samples, observation, *observation_args):
		results = np.array(
			self.trajectory(state, observation_number, 
							steps_per_observation, observation, *observation_args))
		for i in range(samples - 1):
			print("Sample: " + str(i))
			results += np.array(
				self.trajectory(state, observation_number,
								steps_per_observation, observation, *observation_args))
		return results/samples

def expectations(state, observables):
	return np.einsum("j,ijk,k->i", np.conjugate(state).T, observables, state)