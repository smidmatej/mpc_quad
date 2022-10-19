from gp import GPR
import numpy as np
import casadi as cs


class GPEnsemble:

    
    def __init__(self, number_of_dimensions):
        self.gp = [None]*number_of_dimensions
        self.number_of_dimensions = number_of_dimensions
        
    def add_gp(self, new_gp, dim):
        self.gp[dim] = new_gp
        
    def predict(self, z, std=False):
        ### TODO: Add std and variance to casadi prediction ###

        out_j = [None]*self.number_of_dimensions
        if isinstance(z, cs.SX):
            for n in range(len(self.gp)):
                out_j[n] = self.gp[n].predict(cs.reshape(z[:,n],-1,1))
                print(type(out_j[n]))
            concat = [out_j[n] for n in range(len(out_j))]
            #print(len(concat))
            out = cs.horzcat(*concat)
            return out
        else:
            if std:
                # std requested, need to get std from all gps
                std = [None]*self.number_of_dimensions
                for n in range(len(self.gp)):
                    out_j[n], std[n] = self.gp[n].predict(z[:,n].reshape(-1,1), std=True)
                out = np.concatenate(out_j, axis=1)
                return out, std
            else:
                # Nobody wants std
                for n in range(len(self.gp)):
                    out_j[n] = self.gp[n].predict(z[:,n].reshape(-1,1))
                out = np.concatenate(out_j, axis=1)
                return out
            
        
    
    def fit(self):
        for gpr in self.gp:
            gpr.fit()

    def jacobian(self, z):
        """
        Casadi symbolic jacobian of expression self.prediction with respect to z

        :param: z: Casadi symbolic vector expression n x d
        :return: Casadi function jacobian
        """
        assert z.shape[1] == self.number_of_dimensions, f"z needs to be n x d,  z.shape={z.shape}, GPE.number_of_dimensions={self.number_of_dimensions}"

        f_jacobs = list()
        for col in range(self.number_of_dimensions):
            f_jacobs.append(self.gp[col].jacobian(z[:,col]))
        return f_jacobs