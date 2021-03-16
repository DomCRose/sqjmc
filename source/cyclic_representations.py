import math
import numpy as np
import scipy.linalg
from scipy import sparse as sp

class spin_1_number_conserved(object):

	"""Constructs symmetry eigenspaces and projection operators.

	For translation invariance with number conservation.
	"""

	def __init__(self, sites):
		self.sites = sites
		self.ternary_components = 3**(np.arange(self.sites)[::-1])
		self._state_eigenspaces()

	def ternary_conversion(self, number):
		"""Converts a number into its ternary basis vector."""
		state = np.zeros(self.sites)
		for site in range(self.sites):
			remainder = number % 3
			number = math.floor(number / 3)
			state[self.sites - site - 1] = remainder
		return state

	def _cycles(self):
		"""Constructs invariant sets of spin states under translation."""
		spin_states = [N for N in range(3**self.sites)][::-1]
		state_cycles = [[] for i in range(2*self.sites + 1)]
		while len(spin_states) != 0:
			current_cycle = []
			highest_label = spin_states.pop(0)
			ternary_rep = self.ternary_conversion(highest_label)
			magnetization = int(np.sum(ternary_rep))
			current_cycle.append(highest_label)
			next_ternary = np.roll(ternary_rep, 1)
			next_label = int(next_ternary @ self.ternary_components)
			while next_label != highest_label:
				spin_states.remove(next_label)
				current_cycle.append(next_label)
				next_ternary = np.roll(next_ternary, 1)
				next_label = int(next_ternary @ self.ternary_components)
			state_cycles[magnetization].append(current_cycle)
		return state_cycles

	def _state_eigenspaces(self):
		"""Constructs the translation eigenspace projection operators."""
		state_cycles = self._cycles()
		eigenspaces = [[[] for i in range(self.sites)] for i in range(2*self.sites + 1)]
		for magnetization in range(2*self.sites + 1):
			for cycle in state_cycles[magnetization]:
				for vector_index in range(len(cycle)):
					vector = np.zeros((3**self.sites), complex)
					for cycle_index in range(len(cycle)):
						spin_index = 3**self.sites - 1 - cycle[cycle_index]
						vector[spin_index] += np.round(
							np.exp(complex(2*np.pi*cycle_index*vector_index*1j)
								/ complex(len(cycle)))/complex(np.sqrt(len(cycle))),8)
					eigenspaces[magnetization][
						int(vector_index*self.sites/len(cycle))].append(vector)
		self.projectors = [[] for i in range(2*self.sites + 1)]
		self.eigenspace_dimensions = [[] for i in range(2*self.sites + 1)]
		for magnetization in range(2*self.sites + 1):
			for eigenspace in eigenspaces[magnetization]:
				self.projectors[magnetization].append(np.array(eigenspace))
				self.eigenspace_dimensions[magnetization].append(len(eigenspace))

	def eigenspace_labels(self):
		eigenspaces = [(0,0)]
		for magnetization in range(1, 2*self.sites):
			for quasi_momentum in range(self.sites):
				eigenspaces.append((magnetization, quasi_momentum))
		eigenspaces.append((2*self.sites, 0))
		return eigenspaces

	def project_diagonal(self, operator):
		"""Projects out the diagonal blocks of an operator."""
		projected_operator = []
		projector = self.projectors[0][0]
		projected_operator.append(np.around(
			projector @ operator @ np.conjugate(projector).T, 15))
		for magnetization in range(1, 2*self.sites):
			for quasi_momentum in range(self.sites):
				projector = self.projectors[magnetization][quasi_momentum]
				projected_operator.append(np.around(
					projector @ operator @ np.conjugate(projector).T, 15))
		projector = self.projectors[2*self.sites][0]
		projected_operator.append(np.around(
			projector @ operator @ np.conjugate(projector).T, 15))
		return projected_operator

	def project_offset_diagonal(self, operator, mag_offset, momentum_offset):
		"""Projects out the blocks of an offset diagonal for an operator."""
		projected_operator = []
		domain_space = []
		range_space = []
		if mag_offset == 1:
			right_projector = np.conjugate(self.projectors[0][0]).T
			left_projector = self.projectors[mag_offset][momentum_offset]
			projected_operator.append(np.around(left_projector @ operator 
												@ right_projector, 15))
			domain_space.append((0, 0))
			range_space.append((mag_offset, momentum_offset % self.sites))
		if mag_offset == 0 and momentum_offset == 0:
			right_projector = np.conjugate(self.projectors[0][0]).T
			left_projector = self.projectors[0][0]
			projected_operator.append(np.around(left_projector @ operator 
												@ right_projector, 15))
			domain_space.append((0, 0))
			range_space.append((0, 0))
		for magnetization in range(1, 2*self.sites):
			left_magnetization = magnetization + mag_offset
			if left_magnetization == 0:
				right_momentum = (self.sites - momentum_offset) % self.sites
				right_projector = np.conjugate(
					self.projectors[magnetization][right_momentum]).T
				left_projector = self.projectors[0][0]
				projected_operator.append(np.around(left_projector @ operator 
													@ right_projector, 15))
				domain_space.append((magnetization, right_momentum))
				range_space.append((0, 0))
			elif 0 < left_magnetization < 2*self.sites:
				for quasi_momentum in range(self.sites):
					right_projector = np.conjugate(
						self.projectors[magnetization][quasi_momentum]).T
					left_momentum = (quasi_momentum + momentum_offset) % self.sites
					left_projector = self.projectors[left_magnetization][
													 left_momentum]
					projected_operator.append(np.around(left_projector @ operator 
														@ right_projector, 15))
					domain_space.append((magnetization, quasi_momentum))
					range_space.append((left_magnetization, left_momentum))
			elif left_magnetization == 2*self.sites:
				right_momentum = (self.sites - momentum_offset) % self.sites
				right_projector = np.conjugate(
					self.projectors[magnetization][right_momentum]).T
				left_projector = self.projectors[left_magnetization][0]
				projected_operator.append(np.around(left_projector @ operator 
													@ right_projector, 15))
				domain_space.append((magnetization, right_momentum))
				range_space.append((left_magnetization, 0))
		if mag_offset == -1:
			right_projector = np.conjugate(self.projectors[2*self.sites][0]).T
			left_projector = self.projectors[2*self.sites + mag_offset][momentum_offset]
			projected_operator.append(np.around(left_projector @ operator 
												@ right_projector, 15))
			domain_space.append((2*self.sites, 0))
			range_space.append((2*self.sites + mag_offset, momentum_offset % self.sites))
		if mag_offset == 0 and momentum_offset == 0:
			right_projector = np.conjugate(self.projectors[2*self.sites][0]).T
			left_projector = self.projectors[2*self.sites][0]
			projected_operator.append(np.around(left_projector @ operator 
												@ right_projector, 15))
			domain_space.append((2*self.sites, 0))
			range_space.append((2*self.sites, 0))
		return (projected_operator, domain_space, range_space)


class spin_1_number_conserved_sparse(object):

	"""Constructs symmetry eigenspaces and projection operators.

	For translation invariance with number conservation. Used sparse
	matrices at intermediate steps.
	"""

	def __init__(self, sites):
		self.sites = sites
		self.ternary_components = 3**(np.arange(self.sites)[::-1])
		self._state_eigenspaces()

	def ternary_conversion(self, number):
		"""Converts a number into its ternary basis vector."""
		state = np.zeros(self.sites)
		for site in range(self.sites):
			remainder = number % 3
			number = math.floor(number / 3)
			state[self.sites - site - 1] = remainder
		return state

	def _cycles(self):
		"""Constructs invariant sets of spin states under translation."""
		spin_states = [N for N in range(3**self.sites)][::-1]
		state_cycles = [[] for i in range(2*self.sites + 1)]
		self.eigenspace_dimensions = [[0 for i in range(self.sites)] 
									  for i in range(2*self.sites + 1)]
		while len(spin_states) != 0:
			current_cycle = []
			highest_label = spin_states.pop(0)
			ternary_rep = self.ternary_conversion(highest_label)
			magnetization = int(np.sum(ternary_rep))
			current_cycle.append(highest_label)
			next_ternary = np.roll(ternary_rep, 1)
			next_label = int(next_ternary @ self.ternary_components)
			while next_label != highest_label:
				spin_states.remove(next_label)
				current_cycle.append(next_label)
				next_ternary = np.roll(next_ternary, 1)
				next_label = int(next_ternary @ self.ternary_components)
			state_cycles[magnetization].append(current_cycle)
			for momentum_number in range(len(current_cycle)):
				momentum_index = int(momentum_number*self.sites/len(current_cycle))
				self.eigenspace_dimensions[magnetization][momentum_index] += 1
		return state_cycles

	def _state_eigenspaces(self):
		"""Constructs the translation eigenspace projection operators."""
		state_cycles = self._cycles()
		eigenspace_rows = [[[] for i in range(self.sites)] 
					   	   for j in range(2*self.sites + 1)]
		eigenspace_cols = [[[] for i in range(self.sites)] 
						   for j in range(2*self.sites + 1)]
		eigenspace_vals = [[[] for i in range(self.sites)] 
						   for j in range(2*self.sites + 1)]
		eigenspace_indices = [[0 for i in range(self.sites)] 
							  for i in range(2*self.sites + 1)]
		for magnetization in range(2*self.sites + 1):
			for cycle in state_cycles[magnetization]:
				for vector_index in range(len(cycle)):
					momentum = int(vector_index*self.sites/len(cycle))
					eigenspace_index = eigenspace_indices[magnetization][momentum]
					for cycle_index in range(len(cycle)):
						spin_index = 3**self.sites - 1 - cycle[cycle_index]
						eigenspace_rows[magnetization][momentum].append(eigenspace_index)
						eigenspace_cols[magnetization][momentum].append(spin_index)
						eigenspace_vals[magnetization][momentum].append(
							np.round(
								np.exp(complex(2*np.pi*cycle_index*vector_index*1j)
								/ complex(len(cycle)))/complex(np.sqrt(len(cycle))),8))
					eigenspace_indices[magnetization][momentum] += 1
		self.projectors = [[
			sp.csc_matrix(
				(eigenspace_vals[j][i], (eigenspace_rows[j][i], eigenspace_cols[j][i])), 
				shape = (self.eigenspace_dimensions[j][i], 3**self.sites))
				for i in range(self.sites)] 
			for j in range(2*self.sites + 1)]

	def eigenspace_labels(self):
		eigenspaces = [(0,0)]
		for magnetization in range(1, 2*self.sites):
			for quasi_momentum in range(self.sites):
				eigenspaces.append((magnetization, quasi_momentum))
		eigenspaces.append((2*self.sites, 0))
		return eigenspaces

	def project_diagonal(self, operator, sparsity_limit = 0):
		"""Projects out the diagonal blocks of an operator."""
		projected_operator = []
		projector = self.projectors[0][0]
		total_elements = projector.shape[0]**2
		projected_block = projector @ operator @ projector.getH()
		if projected_block.getnnz() > sparsity_limit * total_elements:
			projected_block = projected_block.toarray()
		projected_operator.append(projected_block)
		for magnetization in range(1, 2*self.sites):
			for quasi_momentum in range(self.sites):
				projector = self.projectors[magnetization][quasi_momentum]
				projected_block = projector @ operator @ projector.getH()
				total_elements = projector.shape[0]**2
				if projected_block.getnnz() > sparsity_limit * total_elements:
					projected_block = projected_block.toarray()
				projected_operator.append(projected_block)
		projector = self.projectors[2*self.sites][0]
		total_elements = projector.shape[0]**2
		projected_block = projector @ operator @ projector.getH()
		if projected_block.getnnz() > sparsity_limit * total_elements:
			projected_block = projected_block.toarray()
		projected_operator.append(projected_block)
		return projected_operator

	def project_offset_diagonal(self, operator, mag_offset, momentum_offset,
								sparsity_limit = 0):
		"""Projects out the blocks of an offset diagonal for an operator."""
		projected_operator = []
		domain_space = []
		range_space = []
		if mag_offset == 1:
			right_projector = self.projectors[0][0].getH()
			left_projector = self.projectors[mag_offset][momentum_offset]
			projected_block = left_projector @ operator @ right_projector
			total_elements = left_projector.shape[0] * right_projector.shape[1]
			if projected_block.getnnz() > sparsity_limit * total_elements:
				projected_block = projected_block.toarray()
			projected_operator.append(projected_block)
			domain_space.append((0, 0))
			range_space.append((mag_offset, momentum_offset % self.sites))
		if mag_offset == 0 and momentum_offset == 0:
			right_projector = self.projectors[0][0].getH()
			left_projector = self.projectors[0][0]
			projected_block = left_projector @ operator @ right_projector
			total_elements = left_projector.shape[0] * right_projector.shape[1]
			if projected_block.getnnz() > sparsity_limit * total_elements:
				projected_block = projected_block.toarray()
			projected_operator.append(projected_block)
			domain_space.append((0, 0))
			range_space.append((0, 0))
		for magnetization in range(1, 2*self.sites):
			left_magnetization = magnetization + mag_offset
			if left_magnetization == 0:
				right_momentum = (self.sites - momentum_offset) % self.sites
				right_projector = self.projectors[magnetization][right_momentum].getH()
				left_projector = self.projectors[0][0]
				projected_block = left_projector @ operator @ right_projector
				total_elements = left_projector.shape[0] * right_projector.shape[1]
				if projected_block.getnnz() > sparsity_limit * total_elements:
					projected_block = projected_block.toarray()
				projected_operator.append(projected_block)
				domain_space.append((magnetization, right_momentum))
				range_space.append((0, 0))
			elif 0 < left_magnetization < 2*self.sites:
				for quasi_mom in range(self.sites):
					right_projector = self.projectors[magnetization][quasi_mom].getH()
					left_momentum = (quasi_mom + momentum_offset) % self.sites
					left_projector = self.projectors[left_magnetization][
													 left_momentum]
					projected_block = left_projector @ operator @ right_projector
					total_elements = left_projector.shape[0] * right_projector.shape[1]
					if projected_block.getnnz() > sparsity_limit * total_elements:
						projected_block = projected_block.toarray()
					projected_operator.append(projected_block)
					domain_space.append((magnetization, quasi_mom))
					range_space.append((left_magnetization, left_momentum))
			elif left_magnetization == 2*self.sites:
				right_momentum = (self.sites - momentum_offset) % self.sites
				right_projector = self.projectors[magnetization][right_momentum].getH()
				left_projector = self.projectors[left_magnetization][0]
				projected_block = left_projector @ operator @ right_projector
				total_elements = left_projector.shape[0] * right_projector.shape[1]
				if projected_block.getnnz() > sparsity_limit * total_elements:
					projected_block = projected_block.toarray()
				projected_operator.append(projected_block)
				domain_space.append((magnetization, right_momentum))
				range_space.append((left_magnetization, 0))
		if mag_offset == -1:
			right_projector = self.projectors[2*self.sites][0].getH()
			left_projector = self.projectors[2*self.sites + mag_offset][momentum_offset]
			projected_block = left_projector @ operator @ right_projector
			total_elements = left_projector.shape[0] * right_projector.shape[1]
			if projected_block.getnnz() > sparsity_limit * total_elements:
				projected_block = projected_block.toarray()
			projected_operator.append(projected_block)
			domain_space.append((2*self.sites, 0))
			range_space.append((2*self.sites + mag_offset, momentum_offset % self.sites))
		if mag_offset == 0 and momentum_offset == 0:
			right_projector = self.projectors[2*self.sites][0].getH()
			left_projector = self.projectors[2*self.sites][0]
			projected_block = left_projector @ operator @ right_projector
			total_elements = left_projector.shape[0] * right_projector.shape[1]
			if projected_block.getnnz() > sparsity_limit * total_elements:
				projected_block = projected_block.toarray()
			projected_operator.append(projected_block)
			domain_space.append((2*self.sites, 0))
			range_space.append((2*self.sites, 0))
		return (projected_operator, domain_space, range_space)