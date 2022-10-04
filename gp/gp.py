import numpy as np
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular

class RBF:
    def __init__(self,L=np.eye(10),sigma_f=1):
        self.L = L
        self.sigma_f = sigma_f
        
    def __call__(self, argument_1,argument_2):

        #print(argument_1.shape)
        #print(argument_2.shape)
        difference = argument_1-argument_2
        
        return float(self.sigma_f**2 * np.exp(-1/2*difference.T.dot(np.linalg.inv(self.L*self.L)).dot(difference)))

    def __str__(self):
        return f"L = {self.L}, \n\r Sigma_f = {self.sigma_f}"

def calculate_covariance_matrix(x1,x2,covariance_function):

    cov_mat = np.empty((x1.shape[0], x2.shape[0]))*np.NaN
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    for i in range(x1.shape[0]):
        a = x1[i,:].reshape(-1,1)
        for j in range(x2.shape[0]):
            
            b = x2[j,:].reshape(-1,1)
            #print(a.shape)
            #print(b.shape)
            cov_mat[i,j] = covariance_function(a,b)
            
    return cov_mat
    
    
class GPR:
    def __init__(self,
                 z_train,
                 y_train,
                 covariance_function,
                 noise=0.0):
        self.z_train = z_train
        self.y_train = y_train
        self.covariance_function = covariance_function
        self.noise = noise
        self._memory = None
        
        self.inv_cov_matrix_of_input_data = np.linalg.inv(
            calculate_covariance_matrix(z_train, z_train, covariance_function) \
            + (noise+1e-7)*np.identity(len(z_train)))
        
        print(f'Size of feature training data = {z_train.shape}')
        print(f'Size of output training data = {y_train.shape}')

    def predict(self,at_values_y, var=False, std=False):
        
        sigma_k = calculate_covariance_matrix(self.z_train, at_values_y, self.covariance_function)
        sigma_kk = calculate_covariance_matrix(at_values_y, at_values_y, self.covariance_function)
        
        #print(sigma_k.shape)
        #print(self.inv_cov_matrix_of_input_data.shape)
        #print(self.y_train.shape)
        mean_at_values = sigma_k.T.dot(
                                self.inv_cov_matrix_of_input_data.dot(
                                    self.y_train))
        
        covar_matrix = sigma_kk - sigma_k.T.dot(
                                self.inv_cov_matrix_of_input_data.dot(
                                    sigma_k))


        variance = np.diag(covar_matrix)
        self._memory = {'mean': mean_at_values, 'covariance_matrix': covar_matrix, 'variance':variance}
        
        if var:
            return mean_at_values, variance
        elif std:
            return mean_at_values, np.sqrt(variance)
        else:  
            return mean_at_values
        
        

