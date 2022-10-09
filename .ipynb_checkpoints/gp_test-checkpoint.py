import numpy as np
import matplotlib.pyplot as plt
from gp import *
from data_loader import data_loader
import time


filename = 'trajectory.pkl'
compute_reduction = 100
n_training_samples = 10

d_loader = data_loader(filename, compute_reduction, n_training_samples)               



z = d_loader.get_z(training=False)
y = d_loader.get_y(training=False)[:,0].reshape(-1,1)


z_train = d_loader.get_z(training=True)
y_train = d_loader.get_y(training=True)[:,0].reshape(-1,1)


theta0 = [1,1,1] # Kernel variables
#x_query = np.arange(-20,20,0.05).reshape(
#x_query = np.arange(-20,20,0.05).reshape(-1,1)

model = GPR(z_train, y_train, covariance_function=RBF, theta=theta0)
# Before ML optimization
#mean_b, std_b = model.predict(x_query, std=True)

# Calculate the RMS over all samples
mean_test_before = model.predict(z, std=False)
rms_before = np.sqrt(np.mean((y - mean_test_before)**2))

# ML optimization
model.maximize_likelyhood()

# After optimization
#mean_a, std_a = model.predict(x_query, std=True)

# Calculate the RMS over all samples
mean_test_after = model.predict(z, std=False)
rms_after = np.sqrt(np.mean((y - mean_test_after)**2))

print(f'RMS before = {rms_before}')
print(f'RMS after = {rms_after}')