import numpy as np
import torch
from torch.autograd.functional import jacobian,hessian

def galerkin_coords(point,bins,lims):
    ''' Map a point in R2 to an indicator function.
    

    Parameters
    ----------
    point : array, shape = (2,).
    bins : size of galerkin dictionary, number of indicators used is bins**2.
    lims : array,shape = (2,2), limits for indicator funcitons.

    Returns
    -------
    v : array, shape = (bins,bins), galerkin coordinate.
    basis_edges_x : array, shape = (bins,), x-coords of indicator functions.
    basis_edges_y : array, shape = (bins,), y-coords of indicator functions.
    '''
    v,basis_edges_x,basis_edges_y = np.histogram2d(x = [point[0]],y = [point[1]],bins = [bins,bins], range = lims)
    return v,basis_edges_x,basis_edges_y
    
class PerronFrobenius2D:
    
    def __init__(self,n_basis,lims):
        '''
        

        Parameters
        ----------
        n_basis : int, sqrt(number) of basis functions to use for galerkin method.
        lims : lims : array,shape = (2,2), limits for indicator funcitons in galerkin basis.

        Returns
        -------
        T: array, shape = (n_basis,n_basis), galerkin approximation to PF operator.
        '''
        
        self.n_basis = n_basis
        self.T = np.zeros((n_basis**2,n_basis**2))
        self.lims = lims
        
    
    def estimate(self,trajectory,normalize = True):
        '''
        

        Parameters
        ----------
        trajectory : array, shape = (n_timesteps,2), markovian trajectory.
        
        Returns
        -------
        None.

        '''
        
        for t in range(trajectory.shape[0]-1):
            current_state,next_state = galerkin_coords(trajectory[t],bins=self.n_basis,lims=self.lims)[0],galerkin_coords(trajectory[t+1],bins=self.n_basis,lims=self.lims)[0]
            current_ind = np.argmax(current_state.flatten())
            self.T[:,current_ind] += next_state.flatten()
        
        if normalize:
            self.T /= self.T.sum(axis = 0)
        
    def implied_timescales(self,dt,n_vals, eps = 1e-7):
        '''
        

        Parameters
        ----------
        dt : float, timestep of data used to estimate PF operator.
        n_vals : int, number of eigenvalues to keep.
        eps : optional, to prevent div-by-zero error. The default is 1e-7.

        Returns
        -------
        t_i : array, shape = (n_vals,), implied timescales.

        '''
        
        lamb = np.real(np.linalg.eigvals(self.T))
        idx = lamb.argsort()[::-1]   
        lamb = lamb[idx]
        lamb = lamb[0:n_vals]
        t_i = -(dt/np.log(lamb+eps))
        return lamb,t_i     

def dih_angle(x,first = True):
    ''' Compute dihedral angles for a chain of 5 atoms. 
    
    Parameters
    ----------
    x : tensor, shape = (5,3).
    first : if true, compute the dihedral angle using the first
    four points, otherwise compute it with the last four. The default is True.

    Returns
    -------
    da : tensor, shape = (1,), dihedral angle in (-pi,pi].

    '''
    
    x1,x2,x3,x4 = x[:-1] if first else x[1:] 
    u1 = x1-x2
    u2 = x2-x3
    u3 = x3-x4
    r2 = torch.norm(u2)
    
    y = torch.dot(u2,torch.cross(torch.cross(-u3,u2),torch.cross(u1,u2)))
    x = r2*torch.dot(torch.cross(u1,u2),torch.cross(-u3,u2))
    
    da = torch.atan2(y,x)
    
    return da

def proto_reference_params(x,dV,beta):
    '''
    
    Parameters
    ----------
    x : tensor, shape = (5,3).
    dV : array,shape = (5,3), grad of potential of proto molecular system
    evaluated at x.
    beta : float,inverse temperature parameter.

    Returns
    -------
    y : array, shape = (2,1), effective drift using dihedral angle map.
    z : array, shape = (2,15), effective diffusion using dihedral angle map.
    '''
    
    psidx,psiddx = jacobian(dih_angle,x),hessian(dih_angle,x)
    phidx,phiddx = jacobian(lambda x: dih_angle(x,first=False),x),hessian(lambda x: dih_angle(x,first=False),x)
    
    psidx,psiddx = psidx.detach().cpu().numpy(),psiddx.detach().cpu().numpy()
    phidx,phiddx = phidx.detach().cpu().numpy(),phiddx.detach().cpu().numpy()
    
    
    # compute b^{\xi}
    a = np.zeros((psiddx.shape))
    
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i,j,i,j] = (2/beta)
    
    bpsi,apsi = -np.sum(dV*psidx),np.sum(a*psiddx)
    bphi,aphi = -np.sum(dV*phidx),np.sum(a*phiddx)
    
    y = np.array([[bpsi+0.5*apsi],[bphi+0.5*aphi]])
    
    # compute sigma^{\xi}
    p,d = np.shape(x)
    sigma = np.sqrt(2/beta)*np.eye(int(p*d))
    xi = np.vstack([psidx.flatten(),phidx.flatten()])
    z = np.matmul(xi,sigma)
    
    return y,z

def km_estimators(x_current,x_next,dt):
    ''' Compute Kramer-Moyal estimators at a particular timestep in a trajectory.
    

    Parameters
    ------------
    x_current: array, shape = (d_state,)
    x_next: array, shape = (d_state,)
    dt : float, sampling interval.

    Returns
    -------
    b : empirical drift, array, shape = (1,d_state).
    a : empirical a term, shape = (d_state,d_state).
    '''
    dx = x_next-x_current
    b = dx/dt
    a = (1/dt)*np.outer(dx,dx)
    
    return b,a

def MALA(system,x,dt,beta):
    dim,N = system.dim,system.N
    accepted = False
    while not accepted:
        U,dU = system._potential(x),system._gradient(x)
        pi = np.exp(-beta*U)
        
        w = np.random.randn(N,dim)
        x_prop = x - beta*dU*dt + np.sqrt(2*dt)*w
        
        U_prop,dU_prop = system._potential(x_prop), system._gradient(x_prop)
        pi_prop = np.exp(-beta*U_prop)
        
        q,q_prop = np.exp((-0.25/dt)*((x_prop - x + dt*beta*dU)**2).sum()),np.exp((-0.25/dt)*((x - x_prop + dt*beta*dU_prop)**2).sum())
        
        alpha = (pi_prop/pi)*(q/q_prop)
        
        u = np.random.rand()
        accepted = u < alpha 
        
    return x_prop

def radius_of_gyration(xyz, topology):
    masses = np.array([atom.element.mass for atom in topology.atoms])
    total_mass = masses.sum()
    com = np.array([m*x for m,x in zip(masses,xyz)]).sum(axis = 0) / total_mass
    
    r_sq = ((xyz - com)**2).sum(axis = 1) 
    rog_sq = (masses*r_sq).sum() / total_mass
    
    return np.sqrt(rog_sq)