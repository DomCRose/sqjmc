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
			self.norm = linalg.norm(self.state)
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


	def trajectory(self, steps, state, state_eigenspace):
		self.state = state
		self.state_eigenspace = state_eigenspace
		self.norm = 1
		self.step = 0
		self.steps = steps
		magnetizations = [state_eigenspace[0]]
		momenta = [state_eigenspace[1]]
		obs_steps = [0]
		while self.step < steps:
			self._evolve()
			self._jump()
			magnetizations.append(self.state_eigenspace[0])
			momenta.append(self.state_eigenspace[1])
			obs_steps.append(self.step)
		return magnetizations, momenta, obs_steps


def expectations(state, observables):
	return np.einsum("j,ijk,k->i", np.conjugate(state).T, observables, state)