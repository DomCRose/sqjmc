import math
import numpy as np
from scipy import linalg

class symmetrized_jump_trajectory_generator(object):

	"""Generates quantum jump trajectories within symmetry eigenspaces.


	Tailored to systems with a pair of commuting symmetries. Hamiltonian
	must be provided as a list of blocks for each eigenspace. Each jump
	operator must be provided as three lists, in order this give: a list
	of matrices corresponding to each non-zero block; a list of
	eigenspace labels corresponding to the space each block in the first
	list acts on; a list of eigenspace labels corresponding to the
	eigenspace each block takes the state to. Eigenspaces must be a list 
	of labels corresponding to the eigenspace of each block in the
	Hamiltonian.
	"""

	def __init__(self, model, time_step, eigenspaces):
		self.hamiltonian = model.hamiltonian
		self.jumps = model.jumps
		self.jump_number = len(self.jumps)
		self.eigenspaces = eigenspaces
		self.time_step = time_step
		self._effective_hamiltonian()
		self._evolver()

	def _effective_hamiltonian(self):
		"""Constructs the effective Hamiltonian used to evolve the state."""
		self.effective_hamiltonian = []
		self.activity_op = []
		block_index = 0
		for eigenspace in self.eigenspaces:
			block = self.hamiltonian[block_index]
			dimension = len(block)
			activity_block = np.zeros((dimension, dimension), dtype = complex)
			for jump in self.jumps:
				if eigenspace in jump[1]:
					jump_block = jump[0][jump[1].index(eigenspace)]
					block -= 0.5j*np.conjugate(jump_block).T @ jump_block
					activity_block += np.conjugate(jump_block).T @ jump_block
			block_index += 1
			self.effective_hamiltonian.append(block)
			self.activity_op.append(activity_block)

	def _evolver(self):
		"""Constructs evolution operators within each subspace."""
		self.evolver = []
		for block in self.effective_hamiltonian:
			self.evolver.append(linalg.expm(-1j * self.time_step * block))

	def _evolve(self):
		"""Evolves the state until a jump occurs."""
		jump_time_random = np.random.random()
		evolver = self.evolver[self.eigenspaces.index(self.state_eigenspace)]
		while self.norm > jump_time_random:
			self.state = evolver @ self.state
			self.step += 1
			self.save_step += 1
			self.norm = linalg.norm(self.state)
			if self.save_step == self.save_period:
				self.save_step = 0
				self.magnetizations.append(self.state_eigenspace[0])
				self.momenta.append(self.state_eigenspace[1])
				self.energy.append(
					np.conjugate(self.state).T 
					@ self.hamiltonian[self.eigenspaces.index(self.state_eigenspace)] 
					@ self.state)
				self.activity.append(
					np.conjugate(self.state).T 
					@ self.activity_op[self.eigenspaces.index(self.state_eigenspace)] 
					@ self.state)
			if self.step == self.steps:
				break

	def _jump(self):
		"""Samples a jump operator to apply to the state."""
		jump_probabilities = np.zeros(self.jump_number)
		index = 0
		for jump in self.jumps:
			if self.state_eigenspace in jump[1]:
				jump_state = jump[0][jump[1].index(self.state_eigenspace)] @ self.state
				jump_probabilities[index] = np.conjugate(jump_state).T @ jump_state
			else:
				jump_probabilities[index] = 0
			index += 1
		stacked_probabilities = np.cumsum(jump_probabilities)
		stacked_probabilities /= stacked_probabilities[-1]
		jump_random = np.random.random()
		jump_index = np.searchsorted(stacked_probabilities, jump_random, side = 'right')
		jump = self.jumps[jump_index]
		self.state = (jump[0][jump[1].index(self.state_eigenspace)] 
					  @ self.state)
		self.state /= np.sqrt(jump_probabilities[jump_index])
		self.state_eigenspace = jump[2][jump[1].index(self.state_eigenspace)]
		self.norm = 1


	def trajectory(self, steps, save_period, state, state_eigenspace):
		"""Generates a trajectory sample.
		
		Returns samples of the eigenspace indices, expected energy
		and expected activity rate.
		"""
		self.state = state
		self.state_eigenspace = state_eigenspace
		self.norm = 1
		self.step = 0
		self.steps = steps
		self.save_period = save_period
		self.magnetizations = [state_eigenspace[0]]
		self.momenta = [state_eigenspace[1]]
		self.energy = [
			np.conjugate(self.state).T 
			@ self.hamiltonian[self.eigenspaces.index(self.state_eigenspace)] 
			@ self.state]
		self.activity = [
			np.conjugate(self.state).T 
			@ self.activity_op[self.eigenspaces.index(self.state_eigenspace)] 
			@ self.state]
		self.save_step = 0
		while self.step < steps:
			self._evolve()
			self._jump()
		magnetizations = np.array(self.magnetizations)
		momenta = np.array(self.momenta)
		energy = np.array(self.energy)
		activity = np.array(self.activity)
		return magnetizations, momenta, energy, activity

	def stochastic_average(self, trajectories, steps, save_period, 
						   state, state_eigenspace):
		"""Finds average and variance over trajectory samples.
		
		Returns averages and variances for the eigenspaces indices, 
		expected energy and expected activity rate. Variances are
		calculated as trajectories are generated using Welford's 
		online algorithm https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm.
		"""
		averages = np.zeros((4, math.ceil(steps / save_period) + 1), dtype = complex)
		averages_prior = np.zeros((4, math.ceil(steps / save_period) + 1), 
								  dtype = complex)
		variances = np.zeros((4, math.ceil(steps / save_period) + 1), dtype = complex)
		for i in range(trajectories):
			trajectory_data = self.trajectory(steps, save_period, state, state_eigenspace)
			averages_prior = np.array(averages)
			averages += (trajectory_data - averages)/(i+1)
			variances += ((trajectory_data - averages_prior)
							 * (trajectory_data - averages))
		variances /= (trajectories - 1)
		return np.real(np.concatenate((averages, variances)))