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
model = spin_1.symmetrized_model_sp(sites, interaction_strength, depolarization_strength)

time_step = 0.0005
eigenspaces = model.symmetry_transformer.eigenspace_labels()
print(eigenspaces)
print(model.symmetry_transformer.eigenspace_dimensions)
trajectory_generator = quantum_jumps.symmetrized_jump_trajectory_generator(
	model, time_step, eigenspaces)

samples = 4
steps = 20000
save_period = 10
state = np.array([1])
state_eigenspace = (0,0)
trajectory_data = [[], [], [], []]
colors = []
for i in range(samples):
	initial_time = time.time()
	magnetizations, momenta, energy, activity = trajectory_generator.trajectory(
											steps, save_period, state, state_eigenspace)
	print(time.time() - initial_time)
	trajectory_data[0].append(magnetizations)
	trajectory_data[1].append(momenta)
	trajectory_data[2].append(energy)
	trajectory_data[3].append(activity)
	colors.append(cm.viridis(0.2 + ((0.6 * i) / (samples - 1))))

#np.save("%ssites_%sts_%ssteps_%ssp_mag_mom_energy_activity_samples"%(
#			sites, time_step, steps, save_period),
#		trajectory_data)

times = [time_step * save_period * i for i in range(math.ceil(steps / save_period) + 1)]

trajectories = 100
initial_time = time.time()
data = trajectory_generator.stochastic_average(
	trajectories, steps, save_period, state, state_eigenspace)
print(time.time() - initial_time)
#np.save("%ssites_%sts_%ssteps_%ssp_%strajectories_mag_mom_energy_activity_statistics"%(
#			sites, time_step, steps, save_period, trajectories),
#		data)

axis_pad = 0.15
plt.rc('font', size = 20)
plt.rc('text', usetex = True)
fig = plt.figure(figsize = (9, 5))
ax = plt.subplot(411)
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
ax = plt.subplot(412)
for i in range(samples):
	plt.plot(times, trajectory_data[1][i], color = colors[i], lw = 0.5)
plt.plot(times, data[1], color = cm.viridis(0.3), lw = 2)
plt.fill_between(times, data[1] + np.sqrt(data[4]), data[1] - np.sqrt(data[4]),
				 alpha = 0.5, color = cm.viridis(0.1), lw = 0)
plt.ylim(0 - (sites-1)*axis_pad, sites - 1 + (sites-1)*axis_pad)
plt.yticks([0, sites-1])
plt.ylabel(r'$\left\langle q\right\rangle$')
plt.setp(ax.get_xticklabels(), visible = False)
ax = plt.subplot(413)
for i in range(samples):
	plt.plot(times, trajectory_data[2][i], color = colors[i], lw = 0.5)
plt.plot(times, data[2], color = cm.viridis(0.3), lw = 2)
plt.fill_between(times, data[2] + np.sqrt(data[5]), data[2] - np.sqrt(data[5]),
				 alpha = 0.5, color = cm.viridis(0.1), lw = 0)
plt.ylabel(r'$\left\langle H\right\rangle$')
plt.xlabel(r'$t$', labelpad = -20)
plt.xticks([0, steps * time_step])
plt.ylim(-3 - 6*axis_pad, 3 + 6*axis_pad)
ax = plt.subplot(414)
for i in range(samples):
	plt.plot(times, trajectory_data[3][i], color = colors[i], lw = 0.5)
plt.plot(times, data[3], color = cm.viridis(0.3), lw = 2)
plt.fill_between(times, data[3] + np.sqrt(data[5]), data[2] - np.sqrt(data[5]),
				 alpha = 0.5, color = cm.viridis(0.1), lw = 0)
plt.ylabel(r'$\left\langle H\right\rangle$')
plt.xlabel(r'$t$', labelpad = -20)
plt.xticks([0, steps * time_step])
plt.ylim(-3 - 6*axis_pad, 3 + 6*axis_pad)

plt.subplots_adjust(left = 0.1, right = 0.9, top = 0.9, bottom = 0.1, hspace = 0)
#fig.savefig("trajectories_and_average_T1000_I1_D1.png", dpi = 500)
plt.show()