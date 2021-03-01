import sys
import os
import math
import scipy
import numpy as np
source_path = os.path.join("source/")
sys.path.insert(0, source_path)
import cyclic_representations
source_path = os.path.join("models/")
sys.path.insert(0, source_path)
import spin_1
from matplotlib import pyplot as plt

sites = 4
symmetry_transformer = cyclic_representations.spin_1_number_conserved_representation(
	sites)
print(3**sites)
print(symmetry_transformer.eigenspace_dimensions)
plt.imshow(symmetry_transformer.eigenspace_dimensions)
plt.colorbar()
plt.show()