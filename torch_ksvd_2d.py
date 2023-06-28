from tor_omp import Batch_OMP
import re
import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram
from numpy.linalg import norm
import os
import time
#from numba import jit
import torch


class TorchApproximateKSVD(object):
    def __init__(self, num_basis, coef_sparsity, max_iter=10, tol=1e-6, name = None, logger = None, device = "cuda:0", shouldQ = False, head_size = 0):
        """
        Parameters
        ----------
        num_basis:
            Number of dictionary elements

        max_iter:
            Maximum number of iterations

        tol:
            tolerance for error

        coef_sparsity:
            Number of nonzero coefficients to target
        """
        
        self.basis = None
        self.coefficients = None
        self.max_iter = max_iter
        self.tol = tol
        self.num_basis = num_basis
        self.coef_sparsity = coef_sparsity
        self.name = name
        self.head_size = head_size

        self.device = device
        self.shouldQ = False


    
    def _update_dict_Q(self, X, D, coefficients):

        #? X = (12, vector_num, 64)
        #? D = (12, basis, 64)
        #? coef = (12, vector_num, basis)
        for batch in range(self.batch_size):

            for j in range(self.num_basis):
                I = coefficients[batch, :, j] > 0
                if torch.sum(I) == 0:
                    continue
                
                D[batch,j, :] = 0
                g = coefficients[batch,I, j]
                g = torch.unsqueeze(g, dim=1)
                r = X[batch,I, :] - torch.mm(coefficients[batch,I, :], D[batch])

                #print(r.shape, g.shape)

                d = torch.mm(r.t(),g)
                d = d/torch.norm(d)

                d = torch.squeeze(d)
                
                dQ = self.quantize(d)
                dQ = dQ/torch.norm(dQ)
                
                dQ = torch.unsqueeze(dQ, dim = 1)
                # print(dQ)
                # time.sleep(1000)
                g = torch.mm(r,dQ)                
                dQ = torch.squeeze(dQ)


                D[batch,j, :] = dQ
                coefficients[batch, I, j] = g.t()
            
        return D, None




    def _update_dict(self, X, D, coefficients):

        #? X = (12, vector_num, 64)
        #? D = (12, basis, 64)
        #? coef = (12, vector_num, basis)
        for batch in range(self.batch_size):

            for j in range(self.num_basis):
                I = coefficients[batch, :, j] != 0.0
                if torch.sum(I) == 0:
                    continue
                
                D[batch,j, :] = 0
                g = coefficients[batch,I, j]
                g = torch.unsqueeze(g, dim=1)
                r = X[batch,I, :] - torch.mm(coefficients[batch,I, :], D[batch])

                #print(r.shape, g.shape)

                d = torch.mm(r.t(),g)
                d = d/torch.norm(d) 

                g = torch.mm(r,d)                
                d = torch.squeeze(d)

                D[batch,j, :] = d
                coefficients[batch, I, j] = g.t()
            
        return D, None



    def quantize(self, tensor):
        
        abs_basis = torch.absolute( tensor ).float()
        sign_basis = torch.sign( tensor + 1e-6 )
        m = torch.mean( abs_basis, dim=-1 )
        abs_basis[...,:] = m[...,None]
        basis = abs_basis * sign_basis
        return basis


    def _initialize(self, X):
    
        D = torch.rand(self.batch_size, self.num_basis,self.head_size).to(self.device)
        
        D.requires_grad = False
        D = D/torch.norm(D, dim=2, keepdim=True)

        return D
    
    def _transform(self, D, X):

        # gram = torch.matmul(D,torch.transpose(D, 1,0))
        # Xy = torch.matmul(D, torch.transpose(X,1,0))

        n_nonzero_coefs = self.coef_sparsity

        coef_array = []
        for i in range(self.batch_size):

            #print(X[i].shape, D[i].shape)

            #x_i = torch.transpose(torch.clone(X[i]),1,0)
            #d_i = torch.transpose(torch.clone(D[i]),1,0)
            # print(D.shape)
            # print(X.shape)
            
            coef = Batch_OMP(
                X[i].t(),
                D[i].t(),
                n_nonzero_coefs,
            )
            
            coef_array.append(coef)
        
        coef_final = torch.stack(coef_array, dim=0)
        return coef_final



    def transform_external(self, D, X, n_nonzero_coefs, batch_size):

        # gram = torch.matmul(D,torch.transpose(D, 1,0))
        # Xy = torch.matmul(D, torch.transpose(X,1,0))

        

        coef_array = []
        for i in range(batch_size):

            #print(X[i].shape, D[i].shape)

            #x_i = torch.transpose(torch.clone(X[i]),1,0)
            #d_i = torch.transpose(torch.clone(D[i]),1,0)
            # print(D.shape)
            # print(X.shape)

            coef = Batch_OMP(
                X[i].t(),
                D[i].t(),
                n_nonzero_coefs,
            )
            
            coef_array.append(coef)
        
        coef_final = torch.stack(coef_array, dim=0)
        return coef_final

    def fit(self, X, D_init=None):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """        
        
        self.batch_size = X.shape[0]
        if (D_init is None):
            D = self._initialize(X)
        else:
            D = D_init

        self.org = X
        self.org_norm = torch.norm(self.org, dim=2)
        
        
        for i in range(self.max_iter):
            s_time = time.time()    
            coefficients = self._transform(D, X)
            if (not self.shouldQ):
                D, _ = self._update_dict(X, D, coefficients)
            else:
                D, _ = self._update_dict_Q(X, D, coefficients)
            e_time = time.time()
            print(i, e_time - s_time)

            if (i%20==0 or True):
                self.metric(coefficients, D, i, save = False)
                    
        self.basis = D
        #coefficients = self._transform(D, X) #? remove
        #self.metric(coefficients, self.basis, None, save= True)
        #self.metric(None, self.basis, None, save= True)
        self.metric(coefficients, self.basis, None, save= True)



    
    def fit_external(self, X, D_init=None, coefficients_init = None):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """        
        
        
        self.batch_size = X.shape[0]
        if (D_init is None):
            D = self._initialize(X)
        else:
            D = D_init

        self.org = X
        self.org_norm = torch.norm(self.org, dim=2)
        
        for i in range(self.max_iter):
            s_time = time.time()
            if (i!=0):
                coefficients = self._transform(D, X)
            else:
                if (coefficients_init is None):
                    coefficients = self._transform(D, X)
                else:
                    coefficients = coefficients_init
                
            
            D, _ = self._update_dict(X, D, coefficients)
            
            e_time = time.time()
            print(i, e_time - s_time)

            if (i%5==0 or True):
                self.metric(coefficients, D, i, save = False)
                    
        self.basis = D
        
        self.metric(coefficients, self.basis, None, save= True)



    def transform(self, X):
        return self._transform(self.basis, X)

    def get_frob_norm(self,approx_matrix):

        # ? We need to take the average which is ()
        diff_norm = norm(self.org - approx_matrix, axis=1)

        r = diff_norm/self.org_norm
        r = np.mean(r)
        return r
        
    
    def metric(self,coef,dic, index, save=False):
        
        if (not save):
            error = torch.mean(torch.norm((torch.matmul(coef,dic) - self.org),dim=2)/self.org_norm)
            p = "self.name : " + self.name + " : " + str(error) + " step : " + str(index)
            print(p)
        if (save):
            np_array = dic.detach().cpu().numpy()
            np_coef = coef.detach().cpu().numpy()
            np.save(self.name + ".d.npy", np_array)
            np.save(self.name + ".c.npy", np_coef)
            
