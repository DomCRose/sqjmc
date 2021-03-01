import sys
import os
import numpy as np
import scipy.linalg

class lindbladian(object):

	"""Stores, applies and constructs Lindblad master operators.

	
	"""

	def __init__(self, parameters):
		self.hamiltonian = parameters['hamiltonian']
		self.jump_operators = parameters['jump_operators']
		self.hilbert_space_dimension = len(self.hamiltonian[0])
		self._generate_matrix_representation()

	def _hamiltonian_matrix(self):
		"""Returns a dimensionless hamiltonian matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless Hamiltonians 
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		hamiltonian_term = -1j * np.kron(self.hamiltonian,
										 np.eye(self.hilbert_space_dimension))
		hamiltonian_term += 1j * np.kron(np.eye(self.hilbert_space_dimension),
										 self.hamiltonian.T)
		return hamiltonian_term

	def _jump_term(self, index):
		"""Provides the jump term matrix for the indexed operator."""
		return np.kron(self.jump_operators[index], self.jump_operators[index].conjugate())

	def _jump_matrix(self):
		"""Returns a dimensionless jump group matrix.

		Each term corresponds to a matrix in the choi 
		representation of one of the dimensionless jump operator groups
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		conjugated_jump_operators = np.conjugate(self.jump_operators)
		jump_operator_number = len(conjugated_jump_operators)
		jump_term = self._jump_term(0)
		for i in range(1, jump_operator_number):
			jump_term += self._jump_term(i)
		trace_preservation_term = np.tensordot(conjugated_jump_operators,
											   self.jump_operators,
											   axes = ([0, 1], [0, 1]))
		jump_term -= 0.5*np.kron(trace_preservation_term,
								 np.eye(self.hilbert_space_dimension))
		jump_term -= 0.5*np.kron(np.eye(self.hilbert_space_dimension),
								 trace_preservation_term.T)
		return jump_term

	def action(self, matrix):
		"""Applies the usual action of the Lindblad equation."""
		return matrix

	def adjoint_action(self, matrix):
		"""Applies the adjoint action of the Lindblad equation."""
		return matrix

	def _generate_matrix_representation(self):
		"""Constructs the matrix representation of the Lindblad operator.

		When run for the first time, dimensionless components will be
		saved for future updates if the program has been instructed to.
		"""
		self.matrix_representation = np.zeros((self.hilbert_space_dimension**2,
											   self.hilbert_space_dimension**2), 
											  dtype = complex)
		self.matrix_representation += self._hamiltonian_matrix()
		self.matrix_representation += self._jump_matrix()

	def spectrum(self, return_number = None, extra_eigenvalues = 0, rounding = 10):
		"""Diagonalized the Lindblad matrix and reshapes the eigenvectors."""
		if return_number == None:
			return_number = self.hilbert_space_dimension**2 + 1
		eigenvalues, left_eigenvectors, right_eigenvectors = scipy.linalg.eig(
			self.matrix_representation, left = True)
		sorting_index = eigenvalues.argsort()[::-1]
		eigenvalues = np.around(eigenvalues[sorting_index], rounding)
		left_eigenmatrices = np.reshape(
			left_eigenvectors[:, sorting_index].T, 
			(self.hilbert_space_dimension**2, 
			 self.hilbert_space_dimension, 
			 self.hilbert_space_dimension))
		right_eigenmatrices = np.reshape(
			right_eigenvectors[:, sorting_index].T, 
			(self.hilbert_space_dimension**2, 
			 self.hilbert_space_dimension, 
			 self.hilbert_space_dimension))
		right_eigenmatrices[0] = right_eigenmatrices[0] / np.trace(right_eigenmatrices[0])
		left_eigenmatrices = np.einsum(
			"ijk,i->ijk",
			left_eigenmatrices, 
			1/np.einsum("ikj,ikj->i", 
						np.conjugate(left_eigenmatrices), 
						right_eigenmatrices))
		left_eigenmatrices = np.around(left_eigenmatrices, rounding)
		right_eigenmatrices = np.around(right_eigenmatrices, rounding)
		return (eigenvalues[0 : return_number + extra_eigenvalues], 
				left_eigenmatrices[0 : return_number], 
				right_eigenmatrices[0 : return_number])


class weakly_symmetric_lindbladian(object):

	"""Stores, applies and constructs weakly symmetric Lindbladians.

	Only set up for systems with a single weak symmetry. The jump 
	operators are required to be in a form respecting the weak 
	symmetry, see chapter 3 of the thesis at
	http://eprints.nottingham.ac.uk/56892/ or the paper [REF], input
	as sets of blocks. The Hamiltonian must have also been block 
	diagonalized.
	"""

	def __init__(self, parameters):
		self.hamiltonian = parameters['hamiltonians']
		self.jump_operators = parameters['jump_operators']
		self.eigenspace_pairs = parameters['eigenspace_pairs']
		self.eigenspace_dimensions = parameters['eigenspace_dimensions']
		self.eigenspace_number = len(self.eigenspace_dimensions)
		self._generate_matrix_representation()

	def _hamiltonian_term(self, block_index):
		"""Returns a dimensionless hamiltonian matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless Hamiltonians 
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		hamiltonian_term_blocks = []
		pair_index = self.eigenspace_pairs[block_index][0]
		hamiltonian_term_blocks.append(-1j * np.kron(
			self.hamiltonian[pair_index], np.eye(self.eigenspace_dimensions[0])))
		hamiltonian_term_blocks[-1] += 1j * np.kron(
			np.eye(self.eigenspace_dimensions[pair_index]), self.hamiltonian[0].T)
		for symmetry_index in range(1, self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][symmetry_index]
			hamiltonian_term_blocks.append(-1j * np.kron(
				self.hamiltonian[pair_index], 
				np.eye(self.eigenspace_dimensions[symmetry_index])))
			hamiltonian_term_blocks[-1] += 1j * np.kron(
				np.eye(self.eigenspace_dimensions[pair_index]),
				self.hamiltonian[symmetry_index].T)
		hamiltonian_term = scipy.linalg.block_diag(*hamiltonian_term_blocks)
		return hamiltonian_term

	def _normalization_term(self, block_index):
		conjugated_jumps = np.conjugate(self.jump_operators)
		jump_products = [np.zeros((self.eigenspace_dimensions[i],
								   self.eigenspace_dimensions[i]), dtype = complex) 
						 for i in range(self.eigenspace_number)]
		for jump_index in range(len(self.jump_operators)):
			for symmetry_index in range(self.eigenspace_number):
				jump_products[symmetry_index] += (
					conjugated_jumps[jump_index][symmetry_index].T 
					@ self.jump_operators[jump_index][symmetry_index])
		normalization_term_blocks = []
		pair_index = self.eigenspace_pairs[block_index][0]
		normalization_term_blocks.append(-1j * np.kron(
			jump_products[pair_index], np.eye(self.eigenspace_dimensions[0])))
		normalization_term_blocks[-1] += 1j * np.kron(
			np.eye(self.eigenspace_dimensions[pair_index]), jump_products[0].T)
		for symmetry_index in range(1, self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][symmetry_index]
			normalization_term_blocks.append(-1j * np.kron(
				jump_products[pair_index], 
				np.eye(self.eigenspace_dimensions[symmetry_index])))
			normalization_term_blocks[-1] += 1j * np.kron(
				np.eye(self.eigenspace_dimensions[pair_index]),
				jump_products[symmetry_index].T)
		normalization_term = scipy.linalg.block_diag(*normalization_term_blocks)
		return normalization_term


	def _jump_term(self, block_index):
		"""Returns a dimensionless jump group matrix.

		Each component corresponds to a matrix in the choi 
		representation of one of the dimensionless jump operator groups
		defining the model. These can then be combined linearly with 
		the desired dimensionful parametrization to get the matrix 
		representation of the models full Lindblad operator.
		"""
		adjoint_eigenspace_dimensions = []
		for eigenspace_index in range(self.eigenspace_number):
			pair_index = self.eigenspace_pairs[block_index][eigenspace_index]
			adjoint_eigenspace_dimensions.append(
				self.eigenspace_dimensions[eigenspace_index] 
				* self.eigenspace_dimensions[pair_index])
		jump_term = [[np.zeros((adjoint_eigenspace_dimensions[i],
								adjoint_eigenspace_dimensions[j]), dtype = complex) 
					  for j in range(self.eigenspace_number)]
					 for i in range(self.eigenspace_number)]
		conjugated_jump_operators = np.conjugate(self.jump_operators)
		for jump_index in range(len(self.jump_operators)):
			for symmetry_index in range(self.eigenspace_number):
				pair_index = self.eigenspace_pairs[block_index][symmetry_index]
				print(np.kron(
					self.jump_operators[jump_index][pair_index],
					conjugated_jump_operators[jump_index][symmetry_index]).shape)
				print(jump_term[(symmetry_index - block_index) % 3][symmetry_index].shape)
				jump_term[(symmetry_index - jump_index) % 3][symmetry_index] += np.kron(
					self.jump_operators[jump_index][pair_index],
					conjugated_jump_operators[jump_index][symmetry_index])
		return np.block(jump_term)

	def action(self, matrix):
		"""Applies the usual action of the Lindblad equation."""
		return matrix

	def adjoint_action(self, matrix):
		"""Applies the adjoint action of the Lindblad equation."""
		return matrix

	def _generate_matrix_representation(self):
		"""Constructs the matrix representation of the Lindblad operator.

		When run for the first time, dimensionless components will be
		saved for future updates if the program has been instructed to.
		"""
		self.matrix_representation = []
		for block_index in range(self.eigenspace_number):
			block_dimension = 0
			for eigenspace_index in range(self.eigenspace_number):
				pair_index = self.eigenspace_pairs[block_index][eigenspace_index]
				block_dimension += (self.eigenspace_dimensions[eigenspace_index] 
								* self.eigenspace_dimensions[pair_index])
			self.matrix_representation.append(np.zeros((block_dimension, block_dimension),
											  dtype = complex))
			self.matrix_representation[-1] += self._hamiltonian_term(block_index)
			self.matrix_representation[-1] += self._normalization_term(block_index)
			self.matrix_representation[-1] += self._jump_term(block_index)

	def spectrum(self, return_number = None, extra_eigenvalues = 0, rounding = 10):
		"""Diagonalized the Lindblad matrix and reshapes the eigenvectors."""

		# This needs reworking. Need to loop over blocks, sort, 
		# select a subset, then reshape them into a matrix in the 
		# original basis.
		if return_number == None:
			return_number = self.hilbert_space_dimension**2 + 1
		eigenvalues, left_eigenvectors, right_eigenvectors = scipy.linalg.eig(
			self.matrix_representation, left = True)
		sorting_index = eigenvalues.argsort()[::-1]
		eigenvalues = np.around(eigenvalues[sorting_index], rounding)
		left_eigenmatrices = np.reshape(
			left_eigenvectors[:, sorting_index].T, 
			(self.hilbert_space_dimension**2, 
			 self.hilbert_space_dimension, 
			 self.hilbert_space_dimension))
		right_eigenmatrices = np.reshape(
			right_eigenvectors[:, sorting_index].T, 
			(self.hilbert_space_dimension**2, 
			 self.hilbert_space_dimension, 
			 self.hilbert_space_dimension))
		right_eigenmatrices[0] = right_eigenmatrices[0] / np.trace(right_eigenmatrices[0])
		left_eigenmatrices = np.einsum(
			"ijk,i->ijk",
			left_eigenmatrices, 
			1/np.einsum("ikj,ikj->i", 
						np.conjugate(left_eigenmatrices), 
						right_eigenmatrices))
		left_eigenmatrices = np.around(left_eigenmatrices, rounding)
		right_eigenmatrices = np.around(right_eigenmatrices, rounding)
		return (eigenvalues[0 : return_number + extra_eigenvalues], 
				left_eigenmatrices[0 : return_number], 
				right_eigenmatrices[0 : return_number])