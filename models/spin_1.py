import sys
import os
import math
import numpy as np
from scipy import sparse as sp
source_path = os.path.join("source/")
sys.path.insert(0, source_path)
import cyclic_representations

spin_x = 2**(-0.5)*np.array([[0, 1, 0],
							[1, 0, 1],
							[0, 1, 0]])
spin_y = 2**(-0.5)*np.array([[0, -1j, 0],
							[1j, 0, -1j],
							[0, 1j, 0]])
spin_z = 2**(-0.5)*np.array([[1, 0, 0],
							[0, 0, 0],
							[0, 0, -1]])
spin_up = math.sqrt(0.5) * (spin_x + 1j * spin_y)
spin_down = math.sqrt(0.5) * (spin_x - 1j * spin_y)

def local_operator(operator, index, sites):
	"""Constructs a local operator based on the given operator for a particular site."""
	identity = np.eye(len(operator))
	if index == 1:
		output = np.array(operator)
		for i in range(sites - 1):
			output = np.kron(output, identity)
	else:
		output = identity
		for i in range(index - 2):
			output = np.kron(output, identity)
		output = np.kron(output, np.array(operator))
		for i in range(sites - index):
			output = np.kron(output, identity)
	return output

def local_operator_sp(operator, index, sites):
	"""Constructs a local operator based on the given operator for a particular site."""
	identity = sp.eye(len(operator), format = "csc")
	if index == 1:
		output = sp.csc_matrix(operator)
		for i in range(sites - 1):
			output = sp.kron(output, identity, format = "csc")
	else:
		output = identity
		for i in range(index - 2):
			output = sp.kron(output, identity, format = "csc")
		output = sp.kron(output, sp.csc_matrix(operator), format = "csc")
		for i in range(sites - index):
			output = sp.kron(output, identity, format = "csc")
	return output

def spatial_average(operator, sites):
	"""Constructs an operator giving the spatial average of a local operators."""
	observable = np.zeros((3**sites, 3**sites), dtype = complex)
	for site in range(1, sites + 1):
		observable += local_operator(operator, site, sites)
	return observable/sites


class model(object):

	def __init__(self, spins, interaction_strength, depolarization_strength):
		self.spins = spins
		self.interaction = interaction_strength
		self.depolarization = depolarization_strength
		self.symmetry_transformer = cyclic_representations.spin_1_number_conserved(
			self.spins)
		self._hamiltonian()
		self._jump_operators()

	def _hamiltonian(self):
		self.hamiltonian = np.zeros((3**self.spins, 3**self.spins), dtype = complex)
		left_site_x = local_operator(spin_x, 1, self.spins)
		left_site_y = local_operator(spin_y, 1, self.spins)
		left_site_z = local_operator(spin_z, 1, self.spins)
		for site in range(self.spins):
			next_site = (site + 1) % self.spins
			right_site_x = local_operator(spin_x, next_site + 1, self.spins)
			right_site_y = local_operator(spin_y, next_site + 1, self.spins)
			right_site_z = local_operator(spin_z, next_site + 1, self.spins)
			self.hamiltonian += left_site_x @ right_site_x
			self.hamiltonian += left_site_y @ right_site_y
			self.hamiltonian += left_site_z @ right_site_z
			left_site_x = right_site_x
			left_site_y = right_site_y
			left_site_z = right_site_z
		self.hamiltonian *= self.interaction

	def _jump_operators(self):
		self.jumps = np.zeros((3*self.spins, 3**self.spins, 3**self.spins), dtype = complex)
		for site in range(self.spins):
			self.jumps[site] = local_operator(
				spin_z, site + 1, self.spins)
			self.jumps[site + self.spins] = local_operator(
				spin_up, site + 1, self.spins)
			self.jumps[site + 2*self.spins] = local_operator(
				spin_down, site + 1, self.spins)
		self.jumps *= np.sqrt(self.depolarization)


class model_sp(object):

	def __init__(self, spins, interaction_strength, depolarization_strength):
		self.spins = spins
		self.interaction = interaction_strength
		self.depolarization = depolarization_strength
		self._hamiltonian()
		self._jump_operators()

	def _hamiltonian(self):
		self.hamiltonian = sp.csc_matrix((3**self.spins, 3**self.spins), dtype = complex)
		left_site_x = local_operator_sp(spin_x, 1, self.spins)
		left_site_y = local_operator_sp(spin_y, 1, self.spins)
		left_site_z = local_operator_sp(spin_z, 1, self.spins)
		for site in range(self.spins):
			next_site = (site + 1) % self.spins
			right_site_x = local_operator_sp(spin_x, next_site + 1, self.spins)
			right_site_y = local_operator_sp(spin_y, next_site + 1, self.spins)
			right_site_z = local_operator_sp(spin_z, next_site + 1, self.spins)
			self.hamiltonian += left_site_x @ right_site_x
			self.hamiltonian += left_site_y @ right_site_y
			self.hamiltonian += left_site_z @ right_site_z
			left_site_x = right_site_x
			left_site_y = right_site_y
			left_site_z = right_site_z
		self.hamiltonian.multiply(self.interaction)

	def _jump_operators(self):
		self.jumps = [sp.csc_matrix((3**self.spins, 3**self.spins), dtype = complex)
					  for i in range(3*self.spins)]
		for site in range(self.spins):
			self.jumps[site] = local_operator_sp(
				spin_z, site + 1, self.spins)
			self.jumps[site].multiply(math.sqrt(self.depolarization))
			self.jumps[site + self.spins] = local_operator_sp(
				spin_up, site + 1, self.spins)
			self.jumps[site + self.spins].multiply(math.sqrt(self.depolarization))
			self.jumps[site + 2*self.spins] = local_operator_sp(
				spin_down, site + 1, self.spins)
			self.jumps[site + 2*self.spins].multiply(math.sqrt(self.depolarization))


class symmetrized_model(object):

	def __init__(self, spins, interaction_strength, depolarization_strength):
		self.spins = spins
		self.interaction = interaction_strength
		self.depolarization = depolarization_strength
		self.symmetry_transformer = cyclic_representations.spin_1_number_conserved(
			self.spins)
		self._hamiltonian()
		self._jump_operators()

	def _hamiltonian(self):
		hamiltonian = np.zeros((3**self.spins, 3**self.spins), dtype = complex)
		left_site_x = local_operator(spin_x, 1, self.spins)
		left_site_y = local_operator(spin_y, 1, self.spins)
		left_site_z = local_operator(spin_z, 1, self.spins)
		for site in range(self.spins):
			next_site = (site + 1) % self.spins
			right_site_x = local_operator(spin_x, next_site + 1, self.spins)
			right_site_y = local_operator(spin_y, next_site + 1, self.spins)
			right_site_z = local_operator(spin_z, next_site + 1, self.spins)
			hamiltonian += left_site_x @ right_site_x
			hamiltonian += left_site_y @ right_site_y
			hamiltonian += left_site_z @ right_site_z
			left_site_x = right_site_x
			left_site_y = right_site_y
			left_site_z = right_site_z
		hamiltonian *= self.interaction
		self.hamiltonian = self.symmetry_transformer.project_diagonal(hamiltonian)

	def _jump_operators(self):
		jumps = np.zeros((3*self.spins, 3**self.spins, 3**self.spins), dtype = complex)
		for site in range(self.spins):
			z_op = local_operator(spin_z, site + 1, self.spins)
			for quasi_momentum in range(self.spins):
				jumps[quasi_momentum] += np.exp(
					complex(2*np.pi*site*quasi_momentum*1j) / self.spins) * z_op
			up_op = local_operator(spin_up, site + 1, self.spins)
			for quasi_momentum in range(self.spins):
				jumps[quasi_momentum + self.spins] += np.exp(
					complex(2*np.pi*site*quasi_momentum*1j) / self.spins) * up_op
			down_op = local_operator(spin_down, site + 1, self.spins)
			for quasi_momentum in range(self.spins):
				jumps[quasi_momentum + 2*self.spins] += np.exp(
					complex(2*np.pi*site*quasi_momentum*1j) / self.spins) * down_op
		jumps *= np.sqrt(self.depolarization)
		jumps /= np.sqrt(self.spins)
		self.jumps = []
		for quasi_momentum in range(self.spins):
			self.jumps.append(self.symmetry_transformer.project_offset_diagonal(
				jumps[quasi_momentum], 0, -quasi_momentum))
		for quasi_momentum in range(self.spins):
			self.jumps.append(self.symmetry_transformer.project_offset_diagonal(
				jumps[quasi_momentum + self.spins], 1, -quasi_momentum))
		for quasi_momentum in range(self.spins):
			self.jumps.append(self.symmetry_transformer.project_offset_diagonal(
				jumps[quasi_momentum + 2*self.spins], -1, -quasi_momentum))

class symmetrized_model_sp(object):

	def __init__(self, spins, interaction_strength, depolarization_strength):
		self.spins = spins
		self.interaction = interaction_strength
		self.depolarization = depolarization_strength
		self.symmetry_transformer = cyclic_representations.spin_1_number_conserved_sparse(
			self.spins)
		self._hamiltonian()
		self._jump_operators()

	def _hamiltonian(self):
		hamiltonian = sp.csc_matrix((3**self.spins, 3**self.spins), dtype = complex)
		left_site_x = local_operator_sp(spin_x, 1, self.spins)
		left_site_y = local_operator_sp(spin_y, 1, self.spins)
		left_site_z = local_operator_sp(spin_z, 1, self.spins)
		for site in range(self.spins):
			next_site = (site + 1) % self.spins
			right_site_x = local_operator_sp(spin_x, next_site + 1, self.spins)
			right_site_y = local_operator_sp(spin_y, next_site + 1, self.spins)
			right_site_z = local_operator_sp(spin_z, next_site + 1, self.spins)
			hamiltonian += left_site_x @ right_site_x
			hamiltonian += left_site_y @ right_site_y
			hamiltonian += left_site_z @ right_site_z
			left_site_x = right_site_x
			left_site_y = right_site_y
			left_site_z = right_site_z
		hamiltonian *= self.interaction
		self.hamiltonian = self.symmetry_transformer.project_diagonal(hamiltonian)

	def _jump_operators(self):
		jumps = [sp.csc_matrix((3**self.spins, 3**self.spins), dtype = complex) 
				 for i in range(3*self.spins)]
		scaling_factor = math.sqrt(self.depolarization / self.spins)
		for site in range(self.spins):
			z_op = local_operator_sp(spin_z, site + 1, self.spins)
			up_op = local_operator_sp(spin_up, site + 1, self.spins)
			down_op = local_operator_sp(spin_down, site + 1, self.spins)
			for quasi_momentum in range(self.spins):
				coefficient = scaling_factor * np.exp(
					complex(2*math.pi*site*quasi_momentum*1j) / self.spins)
				jumps[quasi_momentum] += coefficient * z_op
				jumps[quasi_momentum + self.spins] += coefficient * up_op
				jumps[quasi_momentum + 2*self.spins] += coefficient * down_op
		self.symmetrized_jump_nnz = [jump.getnnz() for jump in jumps]
		self.jumps = []
		for quasi_momentum in range(self.spins):
			self.jumps.append(self.symmetry_transformer.project_offset_diagonal(
				jumps[quasi_momentum], 0, -quasi_momentum))
		for quasi_momentum in range(self.spins):
			self.jumps.append(self.symmetry_transformer.project_offset_diagonal(
				jumps[quasi_momentum + self.spins], 1, -quasi_momentum))
		for quasi_momentum in range(self.spins):
			self.jumps.append(self.symmetry_transformer.project_offset_diagonal(
				jumps[quasi_momentum + 2*self.spins], -1, -quasi_momentum))