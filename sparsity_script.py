import sys
import os
import time
import math
import scipy
import numpy as np
source_path = os.path.join("source/")
sys.path.insert(0, source_path)
import cyclic_representations
import quantum_jumps
source_path = os.path.join("models/")
sys.path.insert(0, source_path)
import spin_1
from matplotlib import pyplot as plt
from matplotlib import cm

sites = 6
interaction_strength = 1
depolarization_strength = 1
model = spin_1.model(sites, interaction_strength, depolarization_strength)
sym_model = spin_1.symmetrized_model_sp(sites, interaction_strength, depolarization_strength)
symmetry_transformer = cyclic_representations.spin_1_number_conserved(sites)
eigenspace_dims = symmetry_transformer.eigenspace_dimensions
print(np.sum(eigenspace_dims))
print(3**sites)

print(3**(2*sites))
print(np.count_nonzero(model.hamiltonian))
for jump in model.jumps:
	print(np.count_nonzero(jump))



ham_sparsity = np.zeros((2*sites + 1, sites))
i = 1
ham_sparsity[0][0] = np.count_nonzero(sym_model.hamiltonian[0]) /eigenspace_dims[0][0]**2
for mag in range(1, 2*sites):
	for mom in range(sites):
		print(type(sym_model.hamiltonian[i]))
		ham_sparsity[mag][mom] = (np.count_nonzero(sym_model.hamiltonian[i])
								  / eigenspace_dims[mag][mom]**2)
		i += 1
ham_sparsity[2*sites][0] = (np.count_nonzero(sym_model.hamiltonian[-1]) 
							/ eigenspace_dims[2*sites][0]**2)

jump_sparsity = np.zeros((3*sites, 2*sites + 1, sites))
jump_index = 0
for jump in sym_model.jumps:
	for i in range(len(jump[0])):
		mag_index_1 = jump[1][i][0]
		mag_index_2 = jump[2][i][0]
		mom_index_1 = jump[1][i][1]
		mom_index_2 = jump[2][i][1]
		total_dim = (eigenspace_dims[mag_index_1][mom_index_1]
					 * eigenspace_dims[mag_index_2][mom_index_2])
		jump_sparsity[jump_index][mag_index_1][mom_index_1] = (
			np.count_nonzero(jump[0][i]) / total_dim)
	jump_index += 1


plt.rc('font', size = 20)
plt.rc('text', usetex = True)
fig = plt.figure(figsize = (16, 13))
plt.subplot(4, sites, 3)
plt.imshow(eigenspace_dims)
plt.subplot(4, sites, 4)
plt.imshow(ham_sparsity)
plt.colorbar()
for i in range(3*sites):
	plt.subplot(4, sites, i + 1 + sites)
	plt.imshow(jump_sparsity[i])
	plt.colorbar()


plt.subplots_adjust(left = 0.1, right = 0.9, top = 0.95, bottom = 0.1)
#fig.savefig("sparsity_N6.png", dpi = 500)
plt.show()