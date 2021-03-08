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

sites = 8
interaction_strength = 1
depolarization_strength = 1
#initial_time = time.time()
#model = spin_1.model_sp(sites, interaction_strength, depolarization_strength)
#print(time.time() - initial_time)
initial_time = time.time()
sym_model = spin_1.symmetrized_model_sp(sites, interaction_strength, 
										depolarization_strength)
print(time.time() - initial_time)

print(3**sites)
print(sym_model.symmetry_transformer.eigenspace_dimensions)
time_step = 0.0005
trajectory_generator = quantum_jumps.symmetrized_jump_trajectory_generator(
	sym_model, time_step, sym_model.symmetry_transformer.eigenspace_labels())

#for mag in range(2*sites + 1):
#	for mom in range(sites):
#		print(sym_model.symmetry_transformer.projectors[mag][mom].getformat())
#		print(sym_model.symmetry_transformer.projectors[mag][mom].getnnz())
#		print(sym_model.symmetry_transformer.projectors[mag][mom].get_shape())
#
#for block in sym_model.hamiltonian:
#	print(block.getformat())
#	print(block.getnnz())
#	print(block.get_shape())

#i = 0
#for jump in sym_model.jumps:
#	print("####" + str(i) + "####")
#	j = 0
#	for block in jump[0]:
#		print()
#		print(j)
#		print(block.getformat())
#		print(block.getnnz())
#		print(block.get_shape())
#		j += 1
#	i += 1

samples = 4
steps = 20000
save_period = 10
state = np.array([1])
state_eigenspace = (0,0)
trajectory_data = [[], [], []]
colors = []
for i in range(samples):
	initial_time = time.time()
	magnetizations, momenta, energy = trajectory_generator.trajectory(
											steps, save_period, state, state_eigenspace)
	print(time.time() - initial_time)
	trajectory_data[0].append(magnetizations)
	trajectory_data[1].append(momenta)
	trajectory_data[2].append(energy)
	colors.append(cm.viridis(0.2 + ((0.6 * i) / (samples - 1))))

times = [time_step * save_period * i for i in range(math.ceil(steps / save_period) + 1)]
plt.figure(figsize = (9, 4))
plt.subplot(311)
for i in range(samples):
	plt.plot(times, trajectory_data[0][i])
plt.subplot(312)
for i in range(samples):
	plt.plot(times, trajectory_data[1][i])
plt.subplot(313)
for i in range(samples):
	plt.plot(times, trajectory_data[2][i])
plt.show()