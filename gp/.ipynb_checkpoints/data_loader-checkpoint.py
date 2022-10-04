from numpy.random import default_rng
from six.moves import cPickle as pickle #for performance
rng = default_rng()

import numpy as np



def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


class data_loader:
    def __init__(self, filename, compute_reduction, number_of_training_samples):
        self.dictionary = load_dict(filename)
        
        # takes every *compute_reduction* index along first axis 
        self.compute_reduction = compute_reduction
        self.number_of_training_samples = number_of_training_samples
        
        self.p = self.dictionary['p'][1::compute_reduction,:]
        self.q = self.dictionary['q'][1::compute_reduction,:]
        self.v = self.dictionary['v'][1::compute_reduction,:]
        self.w = self.dictionary['w'][1::compute_reduction,:]
        self.u = self.dictionary['u'][1::compute_reduction,:]
        
        assert self.p.shape[0] > number_of_training_samples , f"Not enough samples for requested number of training samples, try reducting the compute_reduction"
        
        self.a_validation = self.dictionary['aero_drag'][1::compute_reduction]
        
        # error in acceleration between measured and predicted is the regressed variable we are trying to estimate
        self.v_pred = self.dictionary['v_pred'][1::compute_reduction]
        self.y = (self.v - self.v_pred)/self.dictionary['dt']
        self.y_x, self.y_y, self.y_z = np.split(self.y, 3, axis = 1) # divides into xyz dimensions
        
        
        self.z = np.concatenate((self.p, self.q, self.v, self.w, self.u),axis=1)
        
        # subsample further into *number_of_training_samples* samples for training
        self._n_sub = int(self.z.shape[0]/number_of_training_samples)
        
        
        
#     @property
#     def z(self):
#         return self.z

#     @property
#     def y(self):
#         return self.y
    
    def get_z(self, training=False):
        if training:
            return self.z[::self._n_sub, :]
        else:
            return self.z[:,:]
        
    def get_y(self, training=False):
        if training:
            return self.y[::self._n_sub, :]
        else:
            return self.y[:,:]
    
    def get_y_x(self, training=False):
        if training:
            return self.y[::self._n_sub, 0].reshape(-1,1)
        else:
            return self.y[:, 0].reshape(-1,1)
        
    def get_y_y(self, training=False):
        if training:
            return self.y[::self._n_sub, 1].reshape(-1,1)
        else:
            return self.y[:, 1].reshape(-1,1)
        
    def get_y_z(self, training=False):
        if training:
            return self.y[::self._n_sub, 2].reshape(-1,1)
        else:
            return self.y[:, 2].reshape(-1,1)
        
        
    def get_training_samples(self):
        return get_z_training(), get_y_training()
        
    
    def shuffle(self):
        # shuffles the data so that we can take a equidistant subsampling without loss of information
        dataset = np.concatenate((self.z, self.y), axis=1) # concatenates along the state axis
        rng.shuffle(dataset)
        self.z = dataset[:,:-self.y.shape[1]]
        self.y = dataset[:,-self.y.shape[1]:]
