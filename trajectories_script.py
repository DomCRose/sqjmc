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

sites = 10
interaction_strength = 1
depolarization_strength = 1
model = spin_1.symmetrized_model_sp(sites, interaction_strength, depolarization_strength)
symmetry_transformer = cyclic_representations.spin_1_number_conserved(sites)
print(symmetry_transformer.eigenspace_dimensions)

time_step = 0.0005
eigenspaces = symmetry_transformer.eigenspace_labels()
print(eigenspaces)
print(symmetry_transformer.eigenspace_dimensions)
trajectory_generator = quantum_jumps.symmetrized_jump_trajectory_generator(
	model, time_step, eigenspaces)

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
#plt.figure(figsize = (9, 4))
#plt.subplot(311)
#plt.plot(times, magnetizations)
#plt.subplot(312)
#plt.plot(times, momenta)
#plt.subplot(313)
#plt.plot(times, energy)
#plt.show()

trajectories = 1000
initial_time = time.time()
data = trajectory_generator.stochastic_average(
	trajectories, steps, save_period, state, state_eigenspace)
print(time.time() - initial_time)

axis_pad = 0.15
plt.rc('font', size = 20)
plt.rc('text', usetex = True)
fig = plt.figure(figsize = (9, 5))
ax = plt.subplot(311)
for i in range(samples):
	plt.plot(times, trajectory_data[0][i] - sites, color = colors[i], lw = 0.5)
plt.plot(times, data[0] - sites, color = cm.viridis(0.3), lw = 2)
plt.fill_between(times, 
				 data[0] + np.sqrt(data[3]) - sites, 
				 data[0] - np.sqrt(data[3]) - sites,
				 alpha = 0.5, color = cm.viridis(0.1), lw = 0)
plt.ylim(-sites - 2*sites*axis_pad, sites + 2*sites*axis_pad)
plt.yticks([-sites, sites])
plt.ylabel(r'$\left\langle S_z\right\rangle$')
plt.setp(ax.get_xticklabels(), visible = False)
ax = plt.subplot(312)
for i in range(samples):
	plt.plot(times, trajectory_data[1][i], color = colors[i], lw = 0.5)
plt.plot(times, data[1], color = cm.viridis(0.3), lw = 2)
plt.fill_between(times, data[1] + np.sqrt(data[4]), data[1] - np.sqrt(data[4]),
				 alpha = 0.5, color = cm.viridis(0.1), lw = 0)
plt.ylim(0 - (sites-1)*axis_pad, sites - 1 + (sites-1)*axis_pad)
plt.yticks([0, sites-1])
plt.ylabel(r'$\left\langle q\right\rangle$')
plt.setp(ax.get_xticklabels(), visible = False)
ax = plt.subplot(313)
for i in range(samples):
	plt.plot(times, trajectory_data[2][i], color = colors[i], lw = 0.5)
plt.plot(times, data[2], color = cm.viridis(0.3), lw = 2)
plt.fill_between(times, data[2] + np.sqrt(data[5]), data[2] - np.sqrt(data[5]),
				 alpha = 0.5, color = cm.viridis(0.1), lw = 0)
plt.ylabel(r'$\left\langle H\right\rangle$')
plt.xlabel(r'$t$', labelpad = -20)
plt.xticks([0, steps * time_step])
plt.ylim(-3 - 6*axis_pad, 3 + 6*axis_pad)

plt.subplots_adjust(left = 0.1, right = 0.9, top = 0.9, bottom = 0.1, hspace = 0)
#fig.savefig("trajectories_and_average_T1000_I1_D1.png", dpi = 500)
plt.show()