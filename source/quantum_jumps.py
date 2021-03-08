import math
import numpy as np
from scipy import linalg

class symmetrized_jump_trajectory_generator(object):

	"""Generates quantum jump trajectories within symmetry eigenspaces.

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
		self.effective_hamiltonian = []
		block_index = 0
		for eigenspace in self.eigenspaces:
			block = self.hamiltonian[block_index]
			for jump in self.jumps:
				if eigenspace in jump[1]:
					jump_block = jump[0][jump[1].index(eigenspace)]
					block -= 0.5j*np.conjugate(jump_block).T @ jump_block
			block_index += 1
			self.effective_hamiltonian.append(block)

	def _evolver(self):
		self.evolver = []
		for block in self.effective_hamiltonian:
			self.evolver.append(linalg.expm(-1j * self.time_step * block))

	def _evolve(self):
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
			if self.step == self.steps:
				break

	def _jump(self):
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
		self.save_step = 0
		while self.step < steps:
			self._evolve()
			self._jump()
		magnetizations = np.array(self.magnetizations)
		momenta = np.array(self.momenta)
		energy = np.array(self.energy)
		return magnetizations, momenta, energy

	def stochastic_average(self, trajectories, steps, save_period, 
						   state, state_eigenspace):
		magnetizations_average = np.zeros(math.ceil(steps / save_period) + 1,
			dtype = complex)
		momenta_average = np.zeros(math.ceil(steps / save_period) + 1,
			dtype = complex)
		energy_average = np.zeros(math.ceil(steps / save_period) + 1,
			dtype = complex)
		magnetizations_average_prior = np.zeros(math.ceil(steps / save_period) + 1,
			dtype = complex)
		momenta_average_prior = np.zeros(math.ceil(steps / save_period) + 1,
			dtype = complex)
		energy_average_prior = np.zeros(math.ceil(steps / save_period) + 1,
			dtype = complex)
		magnetizations_variance = np.zeros(math.ceil(steps / save_period) + 1,
			dtype = complex)
		momenta_variance = np.zeros(math.ceil(steps / save_period) + 1,
			dtype = complex)
		energy_variance = np.zeros(math.ceil(steps / save_period) + 1,
			dtype = complex)
		for i in range(trajectories):
			magnetizations, momenta, energy = self.trajectory(
				steps, save_period, state, state_eigenspace)
			magnetizations_average_prior = np.array(magnetizations_average)
			momenta_average_prior = np.array(momenta_average)
			energy_average_prior = np.array(energy_average)
			magnetizations_average += (magnetizations - magnetizations_average)/(i+1)
			momenta_average += (momenta - momenta_average)/(i+1)
			energy_average += (energy - energy_average)/(i+1)
			magnetizations_variance += ((magnetizations - magnetizations_average_prior)
										* (magnetizations - magnetizations_average))
			momenta_variance += ((momenta - momenta_average_prior)
								 * (momenta - momenta_average))
			energy_variance += ((energy - energy_average_prior)
								* (energy - energy_average))
		magnetizations_variance /= (trajectories - 1)
		momenta_variance /= (trajectories - 1)
		energy_variance /= (trajectories - 1)
		return magnetizations_average, momenta_average, energy_average, magnetizations_variance, momenta_variance, energy_variance


def expectations(state, observables):
	return np.einsum("j,ijk,k->i", np.conjugate(state).T, observables, state)