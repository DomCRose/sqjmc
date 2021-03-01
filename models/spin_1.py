import sys
import os
import math
import numpy as np

spin_x = 2**(-0.5)*np.array([[0, 1, 0],
							[1, 0, 1],
							[0, 1, 0]])
spin_y = 2**(-0.5)*np.array([[0, -1j, 0],
							[1j, 0, -1j],
							[0, 1j, 0]])
spin_z = 2**(-0.5)*np.array([[1, 0, 0],
							[0, 0, 0],
							[0, 0, -1]])

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

def translation_operator(sites):
	pass


class spin_1_model(object):

	def __init__(self, spins, interaction_strength, depolarization_strength):
		self.spins = spins
		self.interaction = interaction_strength
		self.depolarization = depolarization_strength
		self._hamiltonian()
		self._jump_operators()

	def _hamiltonian(self):
		pass

	def _jump_operators(self):
		pass

