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

sizes = [2,3,4,5,6,7,8]
max_dim = []
average_dim = []

interaction_strength = 1
depolarization_strength = 1
for sites in sizes:
	symmetry_transformer = cyclic_representations.spin_1_number_conserved(sites)
	average_dim.append(np.sum(symmetry_transformer.eigenspace_dimensions)
					   / (2*sites**2 + sites - 2*(sites - 1)))
	max_dim.append(np.max(symmetry_transformer.eigenspace_dimensions))

print(average_dim)
print(max_dim)

plt.rc('font', size = 20)
plt.rc('text', usetex = True)
fig = plt.figure(figsize = (9,4))
ax = plt.subplot(121)
plt.plot(sizes, max_dim)
plt.plot(sizes, average_dim)
plt.xlabel(r'$N$')
ax = plt.subplot(122)
plt.plot(sizes, max_dim)
plt.plot(sizes, average_dim)
plt.xlabel(r'$N$')
plt.yscale('log')
plt.xscale('log')

plt.subplots_adjust(left = 0.1, right = 0.9, top = 0.95, bottom = 0.1)
#fig.savefig("size_dependence.png", dpi = 500)
plt.show()