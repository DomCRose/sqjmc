import sys
import os
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

sites = 8
interaction_strength = 1
depolarization_strength = 10
model = spin_1.model(sites, interaction_strength, depolarization_strength)
symmetry_transformer = cyclic_representations.spin_1_number_conserved(sites)
time_step = 0.0001
eigenspaces = symmetry_transformer.eigenspace_labels()
print(eigenspaces)
print(symmetry_transformer.eigenspace_dimensions)
trajectory_generator = quantum_jumps.symmetrized_jump_trajectory_generator(
	model, time_step, eigenspaces)

steps = 10000
state = np.array([1])
state_eigenspace = (0,0)
magnetizations, momenta, obs_steps = trajectory_generator.trajectory(
										steps, state, state_eigenspace)

plt.figure(figsize = (9, 4))
plt.subplot(211)
plt.plot(obs_steps, magnetizations)
plt.subplot(212)
plt.plot(obs_steps, momenta)
plt.show()