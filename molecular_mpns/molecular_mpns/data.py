from torch_geometric.data import Data
import numpy as np
import torch
from itertools import product
from molecular_mpns.math import MALA

class MarkovMolGraph(Data):
    
    def __init__(self,r_current,r_next,Z):
        '''
        Parameters
        ----------
        r_current : current atomic positions, array, shape = (n_atoms,3).
        r_next : atomic positions one time step later, array, shape = (n_atoms,3).
        Z : current atomic features, array, shape = (n_atoms, feat_dim).

        Returns
        -------
        Data object representing fully-connected molecular graph.
        '''
        
        super(MarkovMolGraph,self).__init__()
        
        self.r_current = torch.tensor(r_current,requires_grad = True)
        self.r_next = torch.tensor(r_next)
        self.Z = torch.tensor(Z)
        self.edge_index = self._create_edges(r_current)
        
    def _create_edges(self,r_current):
        '''
        Parameters
        ----------
        r_current : current atomic positions, array or tensor, shape = (n_atoms,3).

        Returns
        -------
        edge_index: fully-connected edge set,tensor, shape = (2,n_edges).
        '''
    
        # fully connected edge set
        edge_index_i = [i for i,j in product(range(r_current.shape[0]),range(r_current.shape[0]))]
        edge_index_j = [j for i,j in product(range(r_current.shape[0]),range(r_current.shape[0]))]
        edge_index = np.vstack([edge_index_i,edge_index_j])
    

        return torch.tensor(edge_index).long()
    
class MolGraph(Data):
    
    def __init__(self,x,V,dV,requires_grad = False):
        super(MolGraph,self).__init__()
        
        self.x = torch.tensor(x,requires_grad = requires_grad)
        self.V = torch.tensor(V)
        self.dV = torch.tensor(dV)
        self.edge_index = self._create_edges(x)
        
    def _create_edges(self,x):
        '''
        Parameters
        ----------
        x : current atomic positions, array or tensor, shape = (n_atoms,state_dim).

        Returns
        -------
        edge_index: fully-connected edge set,tensor, shape = (2,n_edges).
        '''
    
        # fully connected edge set
        edge_index_i = [i for i,j in product(range(x.shape[0]),range(x.shape[0]))]
        edge_index_j = [j for i,j in product(range(x.shape[0]),range(x.shape[0]))]
        edge_index = np.vstack([edge_index_i,edge_index_j])
    

        return torch.tensor(edge_index).long()
    
def MALA_trajectories(system,n_steps,n_trajs,beta,dt,x0,seed = False):
    
    if seed:
        np.random.seed(seed)
        
    N, dim = system.N, system.dim
    train = []

    for n in range(n_trajs):
        x = x0
        short_traj = np.zeros((n_steps,N*dim))
        for t in range(n_steps):
            x = MALA(system,x,dt,beta)
            short_traj[t,:] = x.flatten()
        train.append(short_traj)
            
    return np.vstack(train)

class AlanineDipeptideGraph(Data):
    
    def __init__(self,z,pos):
        '''
        

        Parameters
        ----------
        z : atomic charges, long tensor, shape = (n_atoms,).
        pos : atomic positions, tensor, shape = (n_atoms,3)

        Returns
        -------
        None.

        '''
        super(AlanineDipeptideGraph,self).__init__()
        self.z = z
        self.pos = pos
        




