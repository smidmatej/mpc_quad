import numpy as np
import matplotlib.pyplot as plt
from gp import *
from data_loader import data_loader
import time
from scipy.stats import multivariate_normal

z_train = np.array([0,1,5]).reshape(-1,1)

y_train = np.array([7,2,3]).reshape(-1,1)
z_query = np.arange(0,10,1).reshape(-1,1)

theta0 = [1,1,1] # Kernel variables

model = GPR(z_train, y_train, covariance_function=KernelFunction, theta=theta0)
model_prior = GPR(None, None, covariance_function=KernelFunction, theta=theta0)
mean_prior, std_prior = model_prior.predict(z_query, covar=True)
print(std_prior)

y = multivariate_normal.pdf(z_query, mean=mean_prior.ravel(), cov=std_prior)
print(y)


# Before ML optimization
#mean_b, std_b = model.predict(z_query, std=True)

# ML optimization
#model.maximize_likelyhood()

# After optimization
#mean_a, std_a = model.predict(z_query, std=True)
