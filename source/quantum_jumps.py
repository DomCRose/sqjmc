import numpy as np
from scipy import linalg

class symmetrized_jump_trajectory_generator(object):

	"""Generates quantum jump trajectories within symmetry eigenspaces.

	"""

	def __init__(self, hamiltonian, jumps, time_step, eigenspaces):
		self.hamiltonian = hamiltonian
		self.jumps = jumps
		self.eigenspaces = eigenspaces
		self.time_step = time_step
		self._effective_hamiltonian()
		self._evolver()

	def _effective_hamiltonian(self):
		self.effective_hamiltonian = []
		for eigenspace in self.eigenspaces:
			self.block = self.hamiltonian[eigenspace]
			for jump in self.jumps:
				pass


def expectations(state, observables):
	return np.einsum("j,ijk,k->i", np.conjugate(state).T, observables, state)