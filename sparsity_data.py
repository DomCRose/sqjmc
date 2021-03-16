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
import matplotlib.colors as colors

interaction_strength = 1
depolarization_strength = 1

site_lengths = [2,3,4,5,6,7,8,9,10,11,12]

def effective_hamiltonian(hamiltonian, jumps):
	effective_hamiltonian = hamiltonian
	for jump in jumps:
		effective_hamiltonian -= 0.5j * jump.getH() @ jump
	return effective_hamiltonian

def effective_hamiltonian_block(hamiltonian, jumps, eigenspaces):
	effective_hamiltonian = []
	block_index = 0
	for eigenspace in eigenspaces:
		block = hamiltonian[block_index]
		for jump in jumps:
			if eigenspace in jump[1]:
				jump_block = jump[0][jump[1].index(eigenspace)]
				block -= 0.5j*np.conjugate(jump_block).T @ jump_block
		block_index += 1
		effective_hamiltonian.append(block)
	return effective_hamiltonian

for sites in site_lengths:
	model = spin_1.model_sp(sites, interaction_strength, depolarization_strength)
	sym_model = spin_1.symmetrized_model_sp(sites, interaction_strength, 
											depolarization_strength)
	eigenspace_dims = sym_model.symmetry_transformer.eigenspace_dimensions
	no_sym_data = [model.hamiltonian.getnnz()]
	for jump in model.jumps:
		no_sym_data.append(jump.getnnz())

	eff_ham = effective_hamiltonian(model.hamiltonian, model.jumps)
	no_sym_data.append(eff_ham.getnnz())
	#np.save("%ssites_no_sym_sparsity"%(sites), no_sym_data)

	#np.save("%ssites_fourier_jumps_sparsity"%(sites), sym_model.symmetrized_jump_nnz)

	ham_sparsity = np.zeros((2*sites + 1, sites))
	i = 1
	ham_sparsity[0][0] = sym_model.hamiltonian[0].getnnz() /eigenspace_dims[0][0]**2
	for mag in range(1, 2*sites):
		for mom in range(sites):
			ham_sparsity[mag][mom] = (sym_model.hamiltonian[i].getnnz()
									/ eigenspace_dims[mag][mom]**2)
			i += 1
	ham_sparsity[2*sites][0] = (sym_model.hamiltonian[-1].getnnz() 
								/ eigenspace_dims[2*sites][0]**2)

	#np.save("%ssites_ham_blocks_sparsity"%(sites), ham_sparsity)

	jump_sparsity = np.zeros((3*sites, 2*sites + 1, sites))
	total_dims = np.zeros((3*sites, 2*sites + 1, sites))
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
				jump[0][i].getnnz() / total_dim)
			total_dims[jump_index][mag_index_1][mom_index_1] = total_dim
		jump_index += 1

	#np.save("%ssites_jump_blocks_sparsity"%(sites), jump_sparsity)
	#np.save("%ssites_jump_blocks_dimension"%(sites), total_dims)
	#np.save("%ssites_eigenspace_dims"%(sites), eigenspace_dims)
	
	eigenspaces = sym_model.symmetry_transformer.eigenspace_labels()
	eff_ham = effective_hamiltonian_block(sym_model.hamiltonian, sym_model.jumps, 
										   eigenspaces)
	eff_ham_sparsity = np.zeros((2*sites + 1, sites))
	i = 1
	eff_ham_sparsity[0][0] = eff_ham[0].getnnz() /eigenspace_dims[0][0]**2
	for mag in range(1, 2*sites):
		for mom in range(sites):
			eff_ham_sparsity[mag][mom] = (eff_ham[i].getnnz()
									/ eigenspace_dims[mag][mom]**2)
			i += 1
	eff_ham_sparsity[2*sites][0] = (eff_ham[-1].getnnz() 
								/ eigenspace_dims[2*sites][0]**2)
	#np.save("%ssites_eff_ham_blocks_sparsity"%(sites), eff_ham_sparsity)



plt.rc('font', size = 20)
plt.rc('text', usetex = True)
fig = plt.figure(figsize = (16, 13))
plt.subplot(4, sites, int(sites / 2))
plt.imshow(eigenspace_dims, cmap = cm.PuBu)
plt.colorbar()
plt.subplot(4, sites, int(sites / 2) + 1)
plt.imshow(ham_sparsity, norm=colors.LogNorm(), cmap = cm.PuBu)
plt.colorbar()
for i in range(3*sites):
	plt.subplot(4, sites, i + 1 + sites)
	plt.imshow(jump_sparsity[i], norm=colors.LogNorm(), cmap = cm.PuBu)
	plt.colorbar()



plt.subplots_adjust(left = 0.1, right = 0.9, top = 0.95, bottom = 0.1)
#fig.savefig("sparsity_N6.png", dpi = 500)
plt.show()