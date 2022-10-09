import numpy as np
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

class KernelFunction:
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


    
    
class GPR:
    def __init__(self,
                 z_train,
                 y_train,
                 covariance_function,
                 theta):


        self.initialize(z_train, y_train, covariance_function, theta)

        
    def initialize(self, z_train, y_train, covariance_function, theta):
        """The whole contructor is in this method. Useful because it will be called after ML optimization"""
        
        self.z_train = z_train
        self.y_train = y_train
        self.covariance_function = covariance_function
        
        self.theta=theta
        
        self.kernel = covariance_function(L=np.eye(z_train.shape[1])*theta[0], sigma_f=theta[-2])
        #self.kernel = covariance_function(L=np.diag(theta[:-2]), sigma_f=theta[-2])
        
        self.noise = theta[-1]
        
        self.inv_cov_matrix_of_input_data = np.linalg.inv(
            self.calculate_covariance_matrix(z_train, z_train, self.kernel) \
            + (self.noise+1e-7)*np.identity(len(z_train)))
        
        print(f'Size of feature training data = {z_train.shape}')
        print(f'Size of output training data = {y_train.shape}')


    def predict(self,at_values_z, var=False, std=False):
        
        sigma_k = self.calculate_covariance_matrix(self.z_train, at_values_z, self.kernel)
        sigma_kk = self.calculate_covariance_matrix(at_values_z, at_values_z, self.kernel)
        
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
        
    #def predict_symbolic(self, at_values_z_sym):
        
        
    def maximize_likelyhood(self):
        
        low_bnd = 0.01
        #bnds = tuple([(low_bnd, None) for i in range(self.z_train.shape[1])]) + ((low_bnd, None), (low_bnd, None))
        bnds = ((low_bnd, None), (low_bnd, None), (low_bnd, None))
        print('Maximizing the likelyhood function for GP')
        print(f'Hyperparameters before optimization = {self.theta}')
        
        sol_min = minimize(self.nll, x0=self.theta, method='L-BFGS-B', bounds=bnds)
        theta_star = sol_min.x
        
        # Ammounts to recreating all relevant contents of this class
        self.initialize(self.z_train, self.y_train, self.covariance_function, theta_star)

        print('Optimization done')
        print(f'Hyperparameters after optimization = {self.theta}')
        
    def nll(self, theta):
            # Numerically more stable implementation of Eq. (11) as described
            # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
            # 2.2, Algorithm 2.1.
            
            #breakpoint()
            k = self.covariance_function(L=np.eye(self.z_train.shape[1])*theta[0], sigma_f=theta[-2])
            #k = self.covariance_function(L=np.diag(theta[:-2]), sigma_f=theta[-2])
            
            K = self.calculate_covariance_matrix(self.z_train, self.z_train, k) + \
                    (theta[-1]+1e-7)*np.identity(self.z_train.shape[0])
            L = cholesky(K)

            S1 = solve_triangular(L, self.y_train, lower=True)
            S2 = solve_triangular(L.T, S1, lower=False)
            
            
            neg_log_lklhd = (np.sum(np.log(np.diagonal(L))) + \
                           0.5 * self.y_train.T.dot(S2) + \
                           0.5 * self.z_train.shape[0] * np.log(2*np.pi)).flatten()
            #print(neg_log_lklhd.shape)
            return neg_log_lklhd
        
    @staticmethod
    def calculate_covariance_matrix(x1,x2, kernel):

        cov_mat = np.empty((x1.shape[0], x2.shape[0]))*np.NaN
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)

        for i in range(x1.shape[0]):
            a = x1[i,:].reshape(-1,1)
            for j in range(x2.shape[0]):

                b = x2[j,:].reshape(-1,1)
                #print(a.shape)
                #print(b.shape)
                cov_mat[i,j] = kernel(a,b)

        return cov_mat


