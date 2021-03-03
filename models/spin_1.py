import sys
import os
import math
import numpy as np
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
		jumps *= self.depolarization
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